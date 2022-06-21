"""Anomaly Map Generator for the PatchCore model implementation."""

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

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import GaussianBlur
from omegaconf import ListConfig


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],   # 原图大小 [512, 512]
        sigma: int = 4,
    ) -> None:
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self, patch_scores: torch.Tensor, feature_map_shape: torch.Size) -> torch.Tensor:
        """ 取topk的每个像素最小值([:,0]),上采样到原图尺寸使用高斯模糊绘制热力图
            Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores                 [64*64, 9]
            feature_map_shape (torch.Size): 2-D feature map shape (width, height)   [64, 64]

        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        width, height = feature_map_shape   # 64, 64
        batch_size = len(patch_scores) // (width * height)  # 1

        # 找最小值绘制热力图
        # print(patch_scores[0])      # tensor([0.7414, 1.0792, 1.1504, 1.1571, 1.2120, 1.2858, 1.2905, 1.3153, 1.3586])
        # print(patch_scores[0][0])   # tensor(0.7414)
        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, width, height))                    # [64*64, 9] -> [64*64, 1] -> [1, 1, 64, 64]
        anomaly_map = F.interpolate(anomaly_map, size=(self.input_size[0], self.input_size[1]))     # [1, 1, 512, 512]

        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1   # kernel_size=33

        # 导出时不使用它，自己写
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(self.sigma, self.sigma))
        # print('anomaly_map', anomaly_map.size())                                                  # [1, 1, 512, 512]
        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: torch.Tensor) -> torch.Tensor:
        """计算图片级别分数
            Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores [4096, 9]
        Returns:
            torch.Tensor: Image-level anomaly scores
        """
        # 找最小值([:,0])中的最大值的下标，最大值意味着找最近的值中最大的，意味着找到的值是没法贴近其他的正常值，它这一行作为异常值进行计算置信度
        # print(patch_scores[:, 0].size())                          # [4096]
        # print(patch_scores[:, 0])                                 # tensor([0.7414, 0.8091, 0.6485,  ..., 0.4588, 0.6546, 0.3370])
        # print(patch_scores[:, 0][1051])                           # tensor(1.1790)
        max_scores = torch.argmax(patch_scores[:, 0])               # tensor(1051)  找最小值中的最大值的下标

        # 根据上面的最大值找到这一行（点）的全部数据
        confidence = torch.index_select(patch_scores, 0, max_scores)# [9] tensor([[1.1790, 1.3289, 1.3604, 1.4017, 1.4162, 1.4228, 1.4311, 1.4492, 1.4502]])
        # 1 - softmax
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))     # 0.8814
        # print('score', torch.max(patch_scores[:, 0]))             # tensor(1.1790)
        score = weights * torch.max(patch_scores[:, 0])             # tensor(1.0392)
        # print('score', score)
        return score

    def __call__(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly
        Expects `feature_map_shape` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array, feature_map_shape=feature_map_shape)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        if "feature_map_shape" not in kwargs:
            raise ValueError(f"Expected key `feature_map_shape`. Found {kwargs.keys()}")

        # nearest_neighbors得到的结果 [64*64, 9]
        patch_scores = kwargs["patch_scores"]
        # 特征图大小 [64, 64]
        feature_map_shape = kwargs["feature_map_shape"]

        # 根据topk的最小值绘制像素级别热力图
        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)     # [1, 1, 512, 512]
        # 得到图片级别分数
        anomaly_score = self.compute_anomaly_score(patch_scores)                    # [1]                   tensor(1.0392)
        # print('anomaly_score', anomaly_score)
        return anomaly_map, anomaly_score
