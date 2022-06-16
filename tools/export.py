"""Expoet onnx model
modified from inference.py
"""


import torch
import onnx
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
    parser.add_argument("--image_size", type=int, required=True, help="Image size,randn(1, 3, args.image_size, args.image_size)")
    parser.add_argument("--meta_data", type=Path, required=False, help="Path to JSON file containing the metadata.")
    args = parser.parse_args()

    return args


def export() -> None:
    """
    """
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin) for the openvino models.
    extension = args.weight_path.suffix
    inferencer: Inferencer
    if extension in (".ckpt"):
        module = import_module("anomalib.deploy.inferencers.torch")
        TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
        inferencer = TorchInferencer(config=config, model_source=args.weight_path, meta_data_path=args.meta_data)
    # elif extension in (".onnx", ".bin", ".xml"):
    #     module = import_module("anomalib.deploy.inferencers.openvino")
    #     OpenVINOInferencer = getattr(module, "OpenVINOInferencer")  # pylint: disable=invalid-name
    #     inferencer = OpenVINOInferencer(config=config, path=args.weight_path, meta_data_path=args.meta_data)
    # else:
    #     raise ValueError(
    #         f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
    #         f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
    #     )

    print(type(inferencer.model))                           # anomalib.models.patchcore.lightning_model.PatchcoreLightning
    print(inferencer.model.image_threshold.cpu().value)     # tensor(0.9531)
    print(inferencer.model.pixel_threshold.cpu().value)     # tensor(0.9531)
    print(inferencer.model.min_max.min.cpu())               # tensor(0.0008)
    print(inferencer.model.min_max.max.cpu())               # tensor(1.6897)
    print(type(inferencer.model.model))                     # anomalib.models.patchcore.torch_model.PatchcoreModel
    print(inferencer.model.model.memory_bank.size())        # torch.Size([13107, 384])

    #-----------------------------#
    #   取出PatchcoreLightning
    #   inferencer.model.model和inferencer.model到处结果相同
    #-----------------------------#
    model = inferencer.model

    #-----------------------------#
    #   导出部分
    #   pytorch1.11不支持导出cdist为onnx，不过github上的实现了，
    #   复制github/torch/onnx/symboloc_opset9.py中对应代码到torch/onnx/symboloc_opset11.py,之所以用11是因为F.interpolate在11中才实现
    #-----------------------------#
    onnx_path = "./results/output.onnx"
    input = torch.randn(1, 3, args.image_size, args.image_size)
    # model = model.model
    model.eval()
    torch.onnx.export(model,                    # 保存的模型
                    input,                      # 模型输入
                    onnx_path,                  # 模型保存 (can be a file or file-like object)
                    export_params=True,         # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                    verbose=False,              # 如果为True，则打印一些转换日志，并且onnx模型中会包含doc_string信息
                    opset_version=11,           # ONNX version 值必须等于_onnx_main_opset或在_onnx_stable_opsets之内。具体可在torch/onnx/symbolic_helper.py中找到
                    do_constant_folding=True,   # 是否使用“常量折叠”优化。常量折叠将使用一些算好的常量来优化一些输入全为常量的节点。
                    input_names=["input"],      # 按顺序分配给onnx图的输入节点的名称列表
                    output_names=["output"],    # 按顺序分配给onnx图的输出节点的名称列表
                    )

    #-----------------------------#
    #   onnx部分
    #-----------------------------#
    # 载入onnx模块
    model_ = onnx.load(onnx_path)
    # print(model_)
    # 检查IR是否良好
    try:
        onnx.checker.check_model(model_)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")


if __name__ == "__main__":
    export()
