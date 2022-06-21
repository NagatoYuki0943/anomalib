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
import json


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
    parser.add_argument("--format", type=str, required=False, help="Onnx or torchscript format")
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
        TorchInferencer = getattr(module, "TorchInferencer")    # pylint: disable=invalid-name
        inferencer = TorchInferencer(config=config, model_source=args.weight_path, meta_data_path=args.meta_data)

    # print('*'*100)
    # print(type(inferencer.model))                             # anomalib.models.patchcore.lightning_model.PatchcoreLightning
    # print(type(inferencer.model.model))                       # anomalib.models.patchcore.torch_model.PatchcoreModel

    image_threshold = inferencer.model.image_threshold.cpu().value.item()
    # print(image_threshold)                                    # 1.0792253017425537
    pixel_threshold = inferencer.model.pixel_threshold.cpu().value.item()
    # print(pixel_threshold)                                    # 1.0792253017425537
    min = inferencer.model.min_max.min.cpu().item()
    # print(min)                                                # 0.02163715288043022
    max = inferencer.model.min_max.max.cpu().item()
    # print(max)                                                # 1.7123807668685913
    # print(inferencer.model.model.memory_bank.size())          # torch.Size([13107, 384])
    # print('*'*100)

    #-----------------------------#
    #   保存参数
    #-----------------------------#
    param = {"image_threshold": image_threshold,
             "pixel_threshold": pixel_threshold,
             "min": min,
             "max": max,
             "pred_image_size": args.image_size,
             }
    with open("./results/param.json", mode='w', encoding='utf-8') as f:
        json.dump(param, f)

    #-----------------------------#
    #   取出PatchcoreLightning
    #   inferencer.model.model和inferencer.model
    #-----------------------------#
    model = inferencer.model.model.eval()
    input = torch.randn(1, 3, args.image_size, args.image_size)

    if args.format != 'torchscript':
        #-----------------------------#
        #   导出onnx
        #   pytorch1.11不支持导出cdist为onnx，不过github上的实现了，
        #   复制github/torch/onnx/symboloc_opset9.py中对应代码到torch/onnx/symboloc_opset11.py,之所以用11是因为F.interpolate在11中才实现
        #-----------------------------#
        onnx_path = "./results/output.onnx"
        # model = model.model
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
    else:
        #-----------------------------#
        #   导出torhscript
        #-----------------------------#
        script_path = "./results/output.torchscript"
        with torch.no_grad():
            trace_module = torch.jit.trace(model, input)
        torch.jit.save(trace_module, script_path)

        trace_module1 = torch.jit.load(script_path)
        print(trace_module1(input)[0].size())           # [1, 1, 512, 512]

if __name__ == "__main__":
    export()
