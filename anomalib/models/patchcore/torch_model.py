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
        self.input_size = input_size        # [224, 224]
        self.num_neighbors = num_neighbors

        # 模型返回layer2和layer3的输出
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1, count_include_pad=False)
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
        # 不需要训练
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        #----------------------------#
        #   对多层输出进行3x3平均池化增加感受野
        #   拼接多层输出
        #----------------------------#
        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)   # [B, 384, 28, 28]

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        #----------------------------#
        #   [B, 384, 28, 28] -> [B*28*28, 384]
        #----------------------------#
        embedding = self.reshape_embedding(embedding)   # [B*28*28, 384]      # 代表将图片分为4096个点，每个点都进行错误预测

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
            # [1, 1, 224, 224]  [1]
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

        embeddings = features[self.layers[0]]   # layer2 [B, 128, 28, 28]
        for layer in self.layers[1:]:           # layer3 [B, 256, 14, 14]
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        # print("embeddings:", embeddings.size()) # [B, 384, 28, 28]
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
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size) # [B*28*28, 384]
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

        # Coreset Subsampling   torch.Size([163850, 384])           0.1
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()  # torch.Size([16385, 384])  下采样到0.1倍
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
        print(embedding.size(), self.memory_bank.size())         # [784, 384] [16385, 384]
        distances = my_cdist_p2_v2(embedding, self.memory_bank)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
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

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)         # (m^test,* in the paper)
        # print(max_patches.size())                           # [b]

        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(patch_scores.shape[0]), max_patches]  # s in the paper
        # print(score.size())                                 # [b]
        nn_index = locations[torch.arange(patch_scores.shape[0]), max_patches]  # m^* in the paper
        # print(nn_index.size())                              # [b]
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]
        # print(nn_sample.size())                             # [b, 384]
        _, support_samples = self.nearest_neighbors(nn_sample, n_neighbors=self.num_neighbors)  # N_b(m^*) in the paper

        # 4. Find the distance of the patch features to each of the support samples
        # print(embedding[max_patches].size())                # [b, 384]
        # print(embedding[max_patches].unsqueeze(1).size())   # [b, 1, 384]   **
        # print(support_samples.size())                       # [b, 9]
        # print(self.memory_bank[support_samples].size())     # [b, 9, 384]   **
        distances = my_cdist_p2_v2(embedding[max_patches].unsqueeze(1), self.memory_bank[support_samples])
        # print(distances.size())                             # [b, 1, 9]     **
        # exit()

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # S^* in the paper
        return score


def my_cdist_p2(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可

        比torch.cdist更慢,不过导出onnx更快
        dim=3时第1个维度代表batch,大了之后相比torch.cdist更慢
        https://github.com/openvinotoolkit/anomalib/issues/440#issuecomment-1191184221

        只能处理二维矩阵
    Args:
        x1 (Tensor): [x, z]
        x2 (Tensor): [y, z]

    Returns:
        Tensor: [x, y]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def my_cdist_p2_v1(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可

        可以处理二维或者三维矩阵
    Args:
        x1 (Tensor): [x, z] or [b, x, z]
        x2 (Tensor): [y, z] or [b, y, z]

    Returns:
        Tensor: [x, y] or [b, x, y]
    """
    if x1.dim() == x2.dim() == 2:
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        res = res.clamp_min_(1e-30).sqrt_()
    elif x1.dim() == x2.dim() == 3:
        # batch=1 不循环加速
        if x1.size(0) == 1:
            # x1.squeeze_(0)    # 这样在pytorch中和 squeeze(0) 效果相同,但是导出onnx会导致输入维度直接变为2维的
            # x2.squeeze_(0)
            x1 = x1.squeeze(0)  # [1, a, x] -> [a, x]
            x2 = x2.squeeze(0)  # [1, b, x] -> [b, x]
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            res.unsqueeze_(0)   # [a, b] -> [1, a, b]
        else:
            # batch > 1
            res = []
            for x1_, x2_ in zip(x1, x2):
                x1_norm = x1_.pow(2).sum(dim=-1, keepdim=True)  # [a, x]
                x2_norm = x2_.pow(2).sum(dim=-1, keepdim=True)  # [a, x]
                res_ = torch.addmm(x2_norm.transpose(-2, -1), x1_, x2_.transpose(-2, -1), alpha=-2).add_(x1_norm)
                res_ = res_.clamp_min_(1e-30).sqrt_()
                res.append(res_)
            res = torch.stack(res, dim=0)   # [a, x] -> [b, a, x]
    return res


def my_cdist_p2_v2(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可

        可以处理多维矩阵
    Args:
        x1 (Tensor): [x, z] or [..., x, z]
        x2 (Tensor): [y, z] or [..., y, z]

    Returns:
        Tensor: [x, y] or [..., x, y]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    # res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = x2_norm.transpose(-2, -1) - 2 * x1 @ x2.transpose(-2, -1) + x1_norm
    res = res.clamp_min_(1e-30).sqrt_()
    return res


if __name__ == "__main__":
    x = torch.randn(640, 384)
    y = torch.randn(6400, 384)
    y0 = torch.cdist(x, y)
    y1 = my_cdist_p2(x, y)
    y2 = my_cdist_p2_v1(x, y)
    y3 = my_cdist_p2_v2(x, y)
    print(y0.size())                # [640, 6400]
    print(y0.eq(y1).sum().item())   # 3248923
    print(y1.eq(y2).sum().item())   # 4096000
    print(y2.eq(y3).sum().item())   # 3796318


    x = torch.randn(1, 640, 384)
    y = torch.randn(1, 6400, 384)
    y0 = torch.cdist(x, y)
    try:
        y1 = my_cdist_p2(x, y)
    except:
        print("my_cdist_p2只能处理二维数据")
    y2 = my_cdist_p2_v1(x, y)
    y3 = my_cdist_p2_v2(x, y)
    print(y0.size())                # [1, 640, 6400]
    print(y0.eq(y2).sum().item())   # 3254730
    print(y2.eq(y3).sum().item())   # 3796352
