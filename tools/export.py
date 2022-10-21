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

from anomalib.deploy import TorchInferencer, export


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


def _export() -> None:
    """Export torch model to onnx, torchscript and openvion, onnx and torchscript is default
    """
    args = get_args()

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(config=args.config, model_source=args.weights)
    model = inferencer.model

    # print(type(model))                                # anomalib.models.patchcore.lightning_model.PatchcoreLightning
    # print(type(model.model))                          # anomalib.models.patchcore.torch_model.PatchcoreModel
    # print(inferencer.model.model.memory_bank.size())  # [16385, 384]

    input_size = args.img_size                          # 推理图片大小
    export_root = args.weights.parent.parent            # weights的上一层目录
    openvino = "openvino" if args.openvino else None    # 是否导出openvino
    export_dir = "export"                               # 存储路径在模型路径上层相邻的export文件夹
    export(model, input_size=input_size, export_mode=openvino, export_root=export_root, export_dir=export_dir)


if __name__ == "__main__":
    _export()
