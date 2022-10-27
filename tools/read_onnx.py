import onnx
import onnxruntime as ort
import numpy as np
import time
import os
from statistics import mean

from read_utils import *


print(ort.__version__)
# print("onnxruntime all providers:", ort.get_all_providers())
print("onnxruntime available providers:", ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
print(ort.get_device())
# GPU


def get_onnx_model(onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
    """获取onnxruntime模型

    Args:
        onnx_path (str): 模型路径
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.

    Returns:
        ort.InferenceSession: 模型session
    """
    mode = mode.lower()
    assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."
    print(f"inference with {mode} !")


    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = {
        "cpu":  ['CPUExecutionProvider'],
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        "cuda": [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ],
        # tensorrt
        # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
        # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
        # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
        "tensorrt": [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2147483648,
                    'trt_fp16_enable': False,
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
    }[mode]

    model = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return model


def single(model_path: str, image_path: str, param_dir: str, save_path: str, mode: str="cpu") -> None:
    """预测单张图片

    Args:
        model_path (str):    模型路径
        image_path (str):    图片路径
        param_dir (str):     超参数路径
        save_path (str): 保存图片路径
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
    """
    # 1.读取模型
    onnx_model = get_onnx_model(model_path, mode)

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
    transform = get_transform(infer_height, infer_width, tensor=False)
    x = transform(image=image)
    x = np.expand_dims(x['image'], axis=0)
    # x = np.ones((1, 3, 224, 224))
    x = x.astype(dtype=np.float32)

    # 5.预测得到热力图和概率
    inputs = onnx_model.get_inputs()
    input_name1 = inputs[0].name
    results = onnx_model.run(None, {input_name1: x})
    anomaly_map, pred_score = results
    print("pred_score:", pred_score)    # 3.1183257

    # 6.后处理,归一化热力图和概率,保存图片
    output, pred_score = post(anomaly_map, pred_score, image, meta_data)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885370492935181
    print("infer time:", end - start)
    cv2.imwrite(save_path, output)


def multi(model_path: str, image_dir: str, param_dir: str, save_dir: str=None, mode: str="cpu") -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        image_dir (str):    图片文件夹
        param_dir (str):    超参数路径
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.读取模型
    onnx_model = get_onnx_model(model_path, mode)

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
        transform = get_transform(infer_height, infer_width, tensor=False)
        x = transform(image=image)
        x = np.expand_dims(x['image'], axis=0)
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 7.预测得到热力图和概率
        inputs = onnx_model.get_inputs()
        input_name1 = inputs[0].name
        results = onnx_model.run(None, {input_name1: x})
        anomaly_map, pred_score = results
        print("pred_score:", pred_score)    # 3.1183257

        # 8.后处理,归一化热力图和概率,保存图片
        output, pred_score = post(anomaly_map, pred_score, image, meta_data)
        end = time.time()
        infer_times.append(end - start)

        print("pred_score:", pred_score)    # 0.8885370492935181
        print("infer time:", end - start)

        # 9.保存图片
        if save_dir is not None:
            save_path = os.path.join(save_dir, img)
            cv2.imwrite(save_path, output)

    print("avg infer time: ", mean(infer_times))


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle-cls/optimization/model.onnx"
    param_dir  = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle-cls/onnx_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle-cls/result"
    single(model_path, image_path, param_dir, save_path, mode="cpu")
    # multi(model_path, image_dir, param_dir, save_dir, mode="cpu")
