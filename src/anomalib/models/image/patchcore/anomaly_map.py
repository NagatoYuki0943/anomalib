"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from omegaconf import ListConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        input_size (ListConfig, tuple): Size of the input image.
            The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel.
            Defaults to ``4``.
    """

    def __init__(
        self,
        input_size: ListConfig | tuple, # [224, 224]
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """ 取topk的每个像素最小值([:,0]),上采样到原图尺寸使用高斯模糊绘制热力图
            Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores                 [28*28, 9]

        Returns:
            Tensor: Map of the pixel-level anomaly scores
        """
        # scale_factor代替size
        # scale_factor = [int(self.input_size[0] / patch_scores.shape[-2]), int(self.input_size[1] / patch_scores.shape[-1])]
        # anomaly_map = F.interpolate(patch_scores, scale_factor=scale_factor)
        anomaly_map = F.interpolate(patch_scores, size=(self.input_size[0], self.input_size[1]))
        return self.blur(anomaly_map)

    def forward(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Return anomaly_map and anomaly_score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
            >>> map = anomaly_map_generator(patch_scores=patch_scores)

        Returns:
            Tensor: anomaly_map
        """
        return self.compute_anomaly_map(patch_scores)
