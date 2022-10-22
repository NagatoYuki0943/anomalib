"""openvino图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化, 这些操作可以通过模型指定
"""


from pathlib import Path

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

import time



def get_model(model_path, device='CPU'):
    """获取模型

    Args:
        model_path (str): 模型路径
        device (str):     模型设备, CPU or GPU
    Returns:
        CompileModel: 编译好的模型
    """
    # Step 1. Initialize OpenVINO Runtime core
    core = ov.Core()
    # Step 2. Read a model
    model = core.read_model(str(Path(model_path)))

    # Step 4. Inizialize Preprocessing for the model  openvino数据预处理
    # https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
    # ppp = PrePostProcessor(model)
    # # Specify input image format 设定图片数据类型，形状，通道排布为BGR
    # ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    # #  Specify preprocess pipeline to input image without resizing 预处理：改变类型，转换为RGB，通道归一化
    # ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # # Specify model's input layout 指定模型输入形状
    # ppp.input().model().set_layout(Layout("NCHW"))
    # #  Specify output results format 指定模型输出类型
    # ppp.output().tensor().set_element_type(Type.f32)
    # # Embed above steps in the graph
    # model = ppp.build()
    compiled_model = core.compile_model(model, device)

    return compiled_model


def main():
    #                        yolov5s_openvino_model_quantization
    openvnio_path = "./results/patchcore/mvtec/bottle-cls/optimization/openvino/model.xml"
    compiled_model = get_model(openvnio_path)

    inputs_names = compiled_model.inputs
    outputs_names = compiled_model.outputs
    print(f"inputs_names: {inputs_names}")      # inputs_names: [<ConstOutput: names[input] shape{1,3,224,224} type: f32>]
    print(f"outputs_names: {outputs_names}")    # outputs_names: [<ConstOutput: names[output] shape{1,1,224,224} type: f32>, <ConstOutput: names[278] shape{1} type: f32>]

    infer_request = compiled_model.create_infer_request()

    x = np.ones((1, 3, 224, 224))
    x = x.astype(dtype=np.float32)

    start = time.time()
    # 设置输入
    # infer_request.infer({0: x}) # 两种方式
    outputs = infer_request.infer({inputs_names[0]: x})

    hotmap = outputs[outputs_names[0]]
    score  = outputs[outputs_names[1]]
    print(hotmap.shape)             # (1, 1, 224, 224)
    print(score)                    # [4.0016055]

    end = time.time()
    print("infer time:", end-start) # infer time: 91.52613687515259


if __name__ == '__main__':
    main()
