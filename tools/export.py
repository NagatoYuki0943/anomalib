"""Anomalib Torch Export Script.
reference tools/inference/torch_inference.py

parent
├── weights
│   └── model.ckpt
└── export              <- export here
    ├── model.onnx
    ├── model_gpu.torchscript
    ├── model_cpu.torchscript
    ├── openvino/...
    └── meta_data.json
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import json
import os
import onnx
from onnxsim import simplify

from anomalib.deploy import TorchInferencer, get_model_metadata


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--img_size", type=str, default=[224, 224], required=False,
                        help="infer image size(height, width), like 224 or [224,224] or 224,224")
    parser.add_argument("--openvino", default=False, action="store_true", required=False, help="export openvino")
    args = parser.parse_args()

    args.img_size = eval(args.img_size)
    # 如果宽高为整数,转换为数组
    if isinstance(args.img_size, int):
        args.img_size = [args.img_size, args.img_size]
    # tuple转换为list
    if isinstance(args.img_size, tuple):
        args.img_size = list(args.img_size)

    return args


def export() -> None:
    """Export torch model to onnx, torchscript and openvion, onnx and torchscript is default
    """
    args = get_args()

    # 存储路径在模型路径上层相邻的export文件夹
    export_path = os.path.join(args.weights.parent.parent, "export")
    if not os.path.exists(export_path):
        os.mkdir(export_path)

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(config=args.config, model_source=args.weights)
    model = inferencer.model
    model.eval()
    # print(type(model))                                # anomalib.models.patchcore.lightning_model.PatchcoreLightning
    # print(type(model.model))                          # anomalib.models.patchcore.torch_model.PatchcoreModel
    # print(inferencer.model.model.memory_bank.size())  # [16385, 384]


    # 导出模型需要的推理数据
    x = torch.zeros((1, 3, *args.img_size))


    #-----------------------------------------------------------
    # onnx
    # 先导出onnx,防止导出torchscript之后的onnx的input和model在不同设备
    onnx_path = os.path.join(export_path, "model.onnx")
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
    if args.openvino:
        openvino_export_path = os.path.join(export_path, "openvino")
        optimize_command = "mo --input_model " + str(onnx_path) + " --output_dir " + str(openvino_export_path)
        assert os.system(optimize_command) == 0, "OpenVINO conversion failed"
        print("export openvino success!")


    #-----------------------------------------------------------
    # 存储配置文件
    # Get metadata
    meta_data = get_model_metadata(inferencer.model)
    # Convert metadata from torch
    for key, value in meta_data.items():
        if isinstance(value, torch.Tensor):
            meta_data[key] = value.numpy().tolist()
    meta_data['img_size'] = args.img_size
    # print(meta_data)
    # {
    #  "image_threshold": 1.5760279893875122,
    #  "pixel_threshold": 1.5760279893875122,
    #  "min": 0.10519619286060333,
    #  "max": 3.9824986457824707,
    #  "img_size": [224, 224]
    # }
    with open(os.path.join(export_path,"meta_data.json"), mode='w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    export()
