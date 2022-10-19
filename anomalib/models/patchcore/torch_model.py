"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size        # [512, 512]
        self.num_neighbors = num_neighbors

        # 模型返回layer2和layer3的输出
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        #-----------------------------------------------#
        # register_buffer
        # 该方法的作用是定义一组参数，该组参数的特别之处在于：
        # 模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        #-----------------------------------------------#
        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    def forward(self, input_tensor: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Embedding for training,
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

        batch_size, _, width, height = embedding.shape
        #----------------------------#
        #   [1, 384, 64, 64] -> [64*64, 384]
        #----------------------------#
        embedding = self.reshape_embedding(embedding)   # [64*64, 384]      # 代表将图片分为4096个点，每个点都进行错误预测

        #--------------------------------------------#
        #   训练直接返回，验证则绘制热力图并计算得分
        #--------------------------------------------#
        if self.training:
            output = embedding
        else:
            # 找到的点计算到所有已知点的距离，返回前n个
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            # compute anomaly score
            anomaly_score = self.compute_anomaly_score(patch_scores, locations, embedding)
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)
            # [1, 1, 512, 512]  [1]
            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """ 将backbone的多层输入在通道上拼接,下层向上上采样
            Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]   # layer2 [1, 128, 28, 28]
        for layer in self.layers[1:]:           # layer3 [1, 256, 14, 14]
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        # print("embeddings:", embeddings.size()) # [1, 384, 28, 28]
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

    def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
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

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> Tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        # 代表将图片分为4096个点，每个点都进行错误预测
        print('nearest_neighbors', embedding.size(), self.memory_bank.size())       # [784, 384] [16385, 384] [1, 384] [16385, 384] 384代表每个点的维度(layer2和layer3拼接为384）
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # (m^test,* in the paper)
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(len(patch_scores)), max_patches]  # s in the paper
        nn_index = locations[torch.arange(len(patch_scores)), max_patches]  # m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]
        _, support_samples = self.nearest_neighbors(nn_sample, n_neighbors=self.num_neighbors)  # N_b(m^*) in the paper
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(embedding[max_patches].unsqueeze(1), self.memory_bank[support_samples], p=2.0)
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze()))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # S^* in the paper
        return score
