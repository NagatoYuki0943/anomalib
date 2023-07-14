"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (tuple[int, int], optional): Size of the model input. Defaults to (224, 224).
        backbone (str, optional): Backbone CNN network. Defaults to "wide_resnet50_2".
        layers (list[str], optional): Layers to extract features from the backbone CNN.
            Defaults to ["layer2", "layer3"].
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone. Defaults to True.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.    所有embeddings存储起来下采样的倍率，存储为memory_bank
            Defaults to 0.1.
        num_neighbors (int, optional): Number of nearest neighbors. Defaults to 9.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer2", "layer3"],
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.model: PatchcoreModel = PatchcoreModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[Tensor] = []      # 每次的结果输出

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.

        self.model.feature_extractor.eval()
        embedding = self.model(batch["image"])  # 训练返回 [64*64, 384]  64*64代表一张图片，可能有多张

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)       # 将每一次的输出都返回

    #-------------------------------------------#
    #   训练完成后验证开始前调用
    #   1.将所有的embedding的列表拼接到一起
    #   2.embedding拼接后下采样放到memory_bank中
    #-------------------------------------------#
    def on_validation_start(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit subsampling at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")

        # 1.将所有的embedding的列表拼接到一起
        embeddings = torch.vstack(self.embeddings) # torch.Size([131072, 384])

        # 2.embedding拼接后下采样放到memory_bank中
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename,
                image, label and mask

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        del args, kwargs  # These variables are not used.

        # 根据topk的最小值绘制像素级别热力图, 得到图片级别分数
        # [1, 1, 224, 224]  [1]
        anomaly_maps, anomaly_score = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score

        return batch


class PatchcoreLightning(Patchcore):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            input_size=hparams.model.input_size,    # [224, 224]
            backbone=hparams.model.backbone,        # resnet18
            layers=hparams.model.layers,            # layer2 layer3
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,    # 0.1
            num_neighbors=hparams.model.num_neighbors,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)


if __name__ == "__main__":
    model = Patchcore((224, 224), "resnet18", ["layer2", "layer3"])
    pth_path = r"D:/ai/code/abnormal/anomalib/results/patchcore/mvtec/carpet/run/weights/model.ckpt"
    pth = torch.load(pth_path)
    model.load_state_dict(pth["state_dict"])
