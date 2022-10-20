"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.types import Number
import onnx
from onnxsim import simplify

from anomalib.models.components import AnomalyModule

#-------------------------------------#
#   在模型中读取保存的metadata
#-------------------------------------#
def get_model_metadata(model: AnomalyModule) -> Dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        Dict[str, Tensor]: metadata
    """
    meta_data = {}
    cached_meta_data: Dict[str, Union[Number, Tensor]] = {
        "image_threshold": model.image_threshold.cpu().value.item(),
        "pixel_threshold": model.pixel_threshold.cpu().value.item(),
    }
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            cached_meta_data[key] = value.cpu()
    # Remove undefined values by copying in a new dict
    for key, val in cached_meta_data.items():
        if not np.isinf(val).all():
            meta_data[key] = val
    del cached_meta_data
    return meta_data


def export_convert(
    model: AnomalyModule,
    input_size: Union[List[int], Tuple[int, int]],
    export_mode: str,
    export_path: Optional[Union[str, Path]] = None,
):
    """Export the model to onnx format and convert to OpenVINO IR. Metadata.json is generated regardless of export mode.

    Args:
        model (AnomalyModule): Model to convert.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
        export_path (Union[str, Path]): Path to exported OpenVINO IR.
        export_mode (str): Mode to export onnx or openvino
    """
    height, width = input_size
    # 输入的图片
    x = torch.zeros((1, 3, height, width))

    # 设置eval模式 important!!!
    model.eval()


    #-----------------------------------------------------------
    # onnx
    # 先导出onnx,防止导出torchscript之后的onnx的input和model在不同设备
    onnx_path = os.path.join(str(export_path), "model.onnx")
    torch.onnx.export(
        model.model,
        x.to(model.device),
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    model_ = onnx.load(onnx_path)
    # 简化模型,更好看
    model_simp, check = simplify(model_)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("export onnx success!")


    #-----------------------------------------------------------
    # torchscript 使用 torch.jit.script, 因为模型forward中有判断
    # cuda
    if torch.cuda.is_available():
        script_path = os.path.join(export_path, "model_gpu.torchscript")
        ts = torch.jit.trace(model.model.cuda(), example_inputs=x.cuda())
        torch.jit.save(ts, script_path)
    # cpu
    script_path = os.path.join(export_path, "model_cpu.torchscript")
    ts = torch.jit.trace(model.model.cpu(), example_inputs=x)
    torch.jit.save(ts, script_path)
    print("export torchscript success!")


    #-----------------------------------------------------------
    # openvino
    if export_mode == "openvino":
        openvino_export_path = os.path.join(str(export_path), export_mode)
        optimize_command = "mo --input_model " + str(onnx_path) + " --output_dir " + str(openvino_export_path)
        assert os.system(optimize_command) == 0, "OpenVINO conversion failed"
        print("export openvino success!")


    #-----------------------------------------------------------
    # 配置文件保存在onnx同目录
    with open(Path(export_path) / "meta_data.json", "w", encoding="utf-8") as metadata_file:
        meta_data = get_model_metadata(model)
        # Convert metadata from torch
        for key, value in meta_data.items():
            if isinstance(value, Tensor):
                meta_data[key] = value.numpy().tolist()
        # save infer image size
        meta_data['img_size'] = [height, width]
        json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)
