import os
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type
import numpy as np
import cv2
import time
import os
from statistics import mean

from read_utils import *


def get_openvino_model(model_path: str, CPU: bool=True, openvino_preprocess: bool=False) -> ov.CompiledModel:
    """获取模型

    Args:
        model_path (str): 模型路径
        CPU (bool, optional): 使用CPU或者GPU. Defaults to True.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.

    Returns:
        CompileModel: 编译好的模型
    """
    # 这里乘以255相当于归一化和标准化同时计算
    mean = np.array((0.485, 0.456, 0.406)) * 255
    std  = np.array((0.229, 0.224, 0.225)) * 255

    # Step 1. Initialize OpenVINO Runtime core
    core = ov.Core()
    # Step 2. Read a model
    model = core.read_model(model_path)

    # 使用openvino数据预处理
    if openvino_preprocess:
        # Step 4. Inizialize Preprocessing for the model  openvino数据预处理
        # https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
        # https://blog.csdn.net/sandmangu/article/details/107181289
        # https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
        ppp = PrePostProcessor(model)
        # 设定图片数据类型，形状，通道排布为RGB     input(0) 指的是第0个输入
        ppp.input(0).tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)
        # 预处理: 改变类型,转换为RGB,减去均值,除以标准差(均值和标准差包含了归一化)
        ppp.input(0).preprocess().convert_element_type(Type.f32).mean(mean).scale(std)
        # 指定模型输入形状
        ppp.input(0).model().set_layout(Layout("NCHW"))
        # 指定模型输出类型
        ppp.output(0).tensor().set_element_type(Type.f32)
        ppp.output(1).tensor().set_element_type(Type.f32)
        # Embed above steps in the graph
        model = ppp.build()

    device = 'CPU' if CPU else 'GPU'
    compiled_model = core.compile_model(model, device_name=device)

    return compiled_model


def single(model_path: str, image_path: str, param_dir: str,
            save_path: str, CPU: bool=True, openvino_preprocess: bool=False) -> None:
    """预测单张图片

    Args:
        model_path (str):    模型路径
        image_path (str):    图片路径
        param_dir (str):     超参数路径
        save_path (str): 保存图片路径
        CPU (bool, optional): CPU or GPU. Defaults to True.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
    """
    # 1.读取模型
    compiled_model = get_openvino_model(model_path, CPU, openvino_preprocess)
    inputs = compiled_model.inputs
    outputs = compiled_model.outputs
    # print(f"inputs: {inputs}")      # inputs: [<ConstOutput: names[input] shape{1,3,224,224} type: f32>]
    # print(f"outputs: {outputs}")    # outputs: [<ConstOutput: names[output] shape{1,1,224,224} type: f32>, <ConstOutput: names[278] shape{1} type: f32>]
    # 创建推理请求
    # infer_request = compiled_model.create_infer_request()

    # 2.获取meta_data
    meta_data = get_meta_data(param_dir)

    # 3.打开图片
    image, origin_height, origin_width = load_image(image_path)
    # 推理时使用的图片大小
    infer_height, infer_width = meta_data["infer_size"]
    # 保存原图宽高
    meta_data["image_size"] = [origin_height, origin_width]

    start = time.time()
    # 4.图片预处理
    if openvino_preprocess:
        # 使用openvino数据预处理要缩放图片
        x = cv2.resize(image, (infer_height, infer_width))
    else:
        transform = get_transform(infer_height, infer_width, tensor=False)
        x = transform(image=image)['image']
    x = np.expand_dims(x, axis=0)
    # x = np.ones((1, 3, 224, 224))
    x = x.astype(dtype=np.float32)

    # 5.预测得到热力图和概率
    # 推理 多种方式
    # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
    # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
    # results = infer_request.infer({inputs[0]: x})     # 同样支持list输入
    # results = compiled_model({inputs[0]: x})
    results = compiled_model([x])
    anomaly_map = results[outputs[0]]
    pred_score  = results[outputs[1]]
    print("pred_score:", pred_score)    # 3.1183267

    # 6.后处理,归一化热力图和概率,保存图片
    output, pred_score = post(anomaly_map, pred_score, image, meta_data)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885372877120972
    print("infer time:", end - start)
    cv2.imwrite(save_path, output)


def multi(model_path: str, image_dir: str, param_dir: str,
            save_dir: str, CPU: bool=True, openvino_preprocess: bool=False) -> None:
    """预测多张图片

    Args:
        model_path (str):    模型路径
        image_dir (str):     图片文件夹
        param_dir (str):     超参数路径
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        CPU (bool, optional): CPU or GPU. Defaults to True.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.读取模型
    compiled_model = get_openvino_model(model_path, CPU, openvino_preprocess)
    inputs = compiled_model.inputs
    outputs = compiled_model.outputs
    # print(f"inputs: {inputs}")      # inputs: [<ConstOutput: names[input] shape{1,3,224,224} type: f32>]
    # print(f"outputs: {outputs}")    # outputs: [<ConstOutput: names[output] shape{1,1,224,224} type: f32>, <ConstOutput: names[278] shape{1} type: f32>]
    # 创建推理请求
    # infer_request = compiled_model.create_infer_request()

    # 2.获取meta_data
    meta_data = get_meta_data(param_dir)

    # 3.获取文件夹中图片
    imgs = os.listdir(image_dir)
    imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

    infer_times: list[float] = []
    # 批量推理
    for img in imgs:
        # 4.拼接图片路径
        image_path = os.path.join(image_dir, img);

        # 5.打开图片
        image, origin_height, origin_width = load_image(image_path)
        # 推理时使用的图片大小
        infer_height, infer_width = meta_data["infer_size"]
        # 保存原图宽高
        meta_data["image_size"] = [origin_height, origin_width]

        start = time.time()
        # 6.图片预处理
        if openvino_preprocess:
            # 使用openvino数据预处理要缩放图片
            x = cv2.resize(image, (infer_height, infer_width))
        else:
            transform = get_transform(infer_height, infer_width, tensor=False)
            x = transform(image=image)['image']
        x = np.expand_dims(x, axis=0)
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 7.预测得到热力图和概率
        # 推理 多种方式
        # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
        # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
        # results = infer_request.infer({inputs[0]: x})     # 同样支持list输入
        # results = compiled_model({inputs[0]: x})
        results = compiled_model([x])
        anomaly_map = results[outputs[0]]
        pred_score  = results[outputs[1]]
        print("pred_score:", pred_score)    # 3.1183267

        # 8.后处理,归一化热力图和概率,保存图片
        output, pred_score = post(anomaly_map, pred_score, image, meta_data)
        end = time.time()
        infer_times.append(end - start)

        print("pred_score:", pred_score)    # 0.8885372877120972
        print("infer time:", end - start)

        # 9.保存图片
        if save_dir is not None:
            save_path = os.path.join(save_dir, img)
            cv2.imwrite(save_path, output)

    print("avg infer time: ", mean(infer_times))


if __name__ == '__main__':
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle-cls/optimization/openvino/model.xml"
    param_dir  = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle-cls/openvino_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle-cls/result"
    single(model_path, image_path, param_dir, save_path, CPU=True, openvino_preprocess=True)
    # multi(model_path, image_dir, param_dir, save_dir, CPU=True, openvino_preprocess=True)
