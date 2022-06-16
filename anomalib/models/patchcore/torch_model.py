"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from anomalib.models.components import (
    DynamicBufferModule,
    FeatureExtractor,
    KCenterGreedy,
)
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "wide_resnet50_2",
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.input_size = input_size        # [512, 512]
        self.num_neighbors = num_neighbors

        # 模型返回layer2和layer3的输出
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)  # ['layer2', 'layer3']
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        #-----------------------------------------------#
        # register_buffer
        # 该方法的作用是定义一组参数，该组参数的特别之处在于：
        # 模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        #-----------------------------------------------#
        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        #----------------------------#
        #   得到backhone的多层输出
        #----------------------------#
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        #----------------------------#
        #   对多层输出进行3x3平均池化增加感受野
        #   拼接多层输出
        #----------------------------#
        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)   # [1, 384, 64, 64]

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        #----------------------------#
        #   [1, 384, 64, 64] -> [64*64, 384]
        #----------------------------#
        feature_map_shape = embedding.shape[-2:]        # [64, 64]
        # print('feature_map_shape:', feature_map_shape)
        embedding = self.reshape_embedding(embedding)   # [64*64, 384]      # 代表将图片分为4096个点，每个点都进行错误预测

        #--------------------------------------------#
        #   训练直接返回，验证则绘制热力图并计算得分
        #--------------------------------------------#
        if self.training:
            output = embedding
        else:
            # 找到的点计算到所有已知点的距离，返回前n个
            patch_scores = self.nearest_neighbors(embedding=embedding, n_neighbors=self.num_neighbors)  # [64*64, 9]
            # 根据topk的最小值绘制像素级别热力图, 得到图片级别分数
            # [1, 1, 512, 512]  [1]
            anomaly_map, anomaly_score = self.anomaly_map_generator(
                patch_scores=patch_scores, feature_map_shape=feature_map_shape  # [64*64, 9], [64, 64]
            )
            print('anomaly_map:', anomaly_map.size())   # torch.Size([1, 1, 512, 512])
            print('anomaly_score:', anomaly_score)      # tensor(1.0392)
            # 根据topk的最小值绘制像素级别热力图, 得到图片级别分数
            # [1, 1, 512, 512]  [1]
            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> torch.Tensor:
        """ 将backbone的多层输入在通道上拼接
            Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]   # 第一层
        for layer in self.layers[1:]:           # 后面的层
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        # print("embeddings:", embeddings.size()) # [1, 384, 64, 64]
        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.   [1, 384, 64, 64]

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)  # 384
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size) # [64*64, 384]
        # print('embedding', embedding.size())
        return embedding

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """训练过程中会将所有的类似[64*64, 384]数据存储起来，这里将它下采样到 10%放到memeory_bank
            会在验证之前调用，训练一轮后会调用这个函数
            Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling   torch.Size([131072, 384])           0.1
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()  # torch.Size([13107, 384])  下采样到0.1倍
        # 将下采样1/10的数据存储起来，放到menory_bank中
        self.memory_bank = coreset

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int = 9) -> Tensor:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
        """
        # 代表将图片分为4096个点，每个点都进行错误预测
        print('nearest_neighbors', embedding.size(), self.memory_bank.size())       # [4096, 384] [13107, 384] 384代表每个点的维度(layer2和layer3拼接为384）
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        print('distances:', distances.size())                                       # [4096, 13107] 代表4096个点和默认的13107个点都计算了距离，每一行代表1个点到13107个点的距离
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        print('patch_scores:', patch_scores.size())                                 # [4096, 9] 保留最近的9个点 largest=False
        return patch_scores
