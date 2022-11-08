from sys import meta_path
import onnx
import onnxruntime as ort
import numpy as np
import time
import os
from statistics import mean

from infer import Inference
from read_utils import *


print(ort.__version__)
# print("onnxruntime all providers:", ort.get_all_providers())
print("onnxruntime available providers:", ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
print(ort.get_device())
# GPU


class OrtInference(Inference):
    def __init__(self, model_path: str, meta_path: str, mode: str="cpu") -> None:
        """
        Args:
            model_path (str): 模型路径
            meta_path (str): 超参数路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__()
        # 超参数
        self.meta  = get_meta_data(meta_path)
        # 载入模型
        self.model = self.get_onnx_model(model_path, mode)
        # 预热模型
        self.warm_up()


    def get_onnx_model(self, onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
        """获取onnxruntime模型

        Args:
            onnx_path (str):    模型路径
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
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


    def warm_up(self):
        """预热模型
        """
        # 预热模型
        infer_height, infer_width = self.meta["infer_size"]
        x = np.zeros((1, 3, infer_height, infer_width), dtype=np.float32)
        self.model.run(None, {self.model.get_inputs()[0].name: x})


    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: 热力图和得分
        """
        # 1.保存原图宽高
        self.meta["image_size"] = [image.shape[0], image.shape[1]]

        # 2.图片预处理
        # 推理时使用的图片大小
        infer_height, infer_width = self.meta["infer_size"]
        transform = get_transform(infer_height, infer_width, tensor=False)
        x = transform(image=image)
        x = np.expand_dims(x['image'], axis=0)
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 2.推理
        inputs = self.model.get_inputs()
        input_name1 = inputs[0].name
        results = self.model.run(None, {input_name1: x})
        anomaly_map, pred_score = results
        print("pred_score:", pred_score)    # 3.1183257

        # 3.后处理,归一化热力图和概率,保存图片
        output, pred_score = self.post(anomaly_map, pred_score, image)

        return output, pred_score


def single(model_path: str, image_path: str, meta_path: str, save_path: str, mode: str="cpu") -> None:
    """预测单张图片

    Args:
        model_path (str):   模型路径
        image_path (str):   图片路径
        meta_path (str):    超参数路径
        save_path (str):    保存图片路径
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
    """
    # 1.获取推理器
    inference = OrtInference(model_path, meta_path, mode)

    # 2.打开图片
    image = load_image(image_path)

    # 3.推理
    start = time.time()
    output, pred_score = inference.infer(image)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885370492935181
    print("infer time:", end - start)

    # 4.保存图片
    cv2.imwrite(save_path, output)


def multi(model_path: str, image_dir: str, meta_path: str, save_dir: str=None, mode: str="cpu") -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        image_dir (str):    图片文件夹
        meta_path (str):    超参数路径
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

    # 1.获取推理器
    inference = OrtInference(model_path, meta_path, mode)

    # 2.获取文件夹中图片
    imgs = os.listdir(image_dir)
    imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

    infer_times: list[float] = []
    scores: list[float] = []
    # 批量推理
    for img in imgs:
        # 3.拼接图片路径
        image_path = os.path.join(image_dir, img);

        # 4.打开图片
        image = load_image(image_path)

        # 5.推理
        start = time.time()
        output, pred_score = inference.infer(image)
        end = time.time()

        infer_times.append(end - start)
        scores.append(pred_score)
        print("pred_score:", pred_score)    # 0.8885370492935181
        print("infer time:", end - start)

        # 6.保存图片
        if save_dir is not None:
            save_path = os.path.join(save_dir, img)
            cv2.imwrite(save_path, output)

    print("avg infer time: ", mean(infer_times))
    draw_score(scores, save_dir)


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle-cls/optimization/model.onnx"
    param_dir  = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle-cls/onnx_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle-cls/result"
    single(model_path, image_path, param_dir, save_path, mode="cuda")
    # multi(model_path, image_dir, param_dir, save_dir, mode="cuda")

