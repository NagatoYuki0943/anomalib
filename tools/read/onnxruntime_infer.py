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
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
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
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 2GB
                        'trt_fp16_enable': False,
                    }),
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
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
            tuple[np.ndarray, float]: hotmap, score
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
        predictions = self.model.run(None, {input_name1: x})    # 返回值为list

        # 3.解决不同模型输出问题
        if len(predictions) == 1:
            # 大多数模型只返回热力图
            # https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/deploy/inferencers/torch_inferencer.py#L159
            anomaly_map = predictions[0]                # [1, 1, 256, 256] 返回类型为 np.ndarray
            pred_score  = anomaly_map.reshape(-1).max() # [1]
        else:
            # patchcore返回热力图和得分
            anomaly_map, pred_score = predictions
        print("pred_score:", pred_score)    # 3.1183257

        # 4.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

        return anomaly_map, pred_score


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
    anomaly_map, pred_score = inference.infer(image)    # [900, 900] [1]

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885370492935181
    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, pred_score, image, mask, mask_outline, superimposed_map)


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
        anomaly_map, pred_score = inference.infer(image)    # [900, 900] [1]

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        infer_times.append(end - start)
        scores.append(pred_score)
        print("pred_score:", pred_score)    # 0.8885370492935181
        print("infer time:", end - start)

        # if save_dir is not None:
        #     # 7.保存图片
        #     save_path = os.path.join(save_dir, img)
        #     save_image(save_path, pred_score, image, mask, mask_outline, superimposed_map)

    print("avg infer time: ", mean(infer_times))
    draw_score(scores, save_dir)


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/fastflow/mvtec/bottle/run/optimization/model.onnx"
    meta_path  = "./results/fastflow/mvtec/bottle/run/optimization/meta_data.json"
    save_path  = "./results/fastflow/mvtec/bottle/run/onnx_output.jpg"
    save_dir   = "./results/fastflow/mvtec/bottle/run/result"
    single(model_path, image_path, meta_path, save_path, mode="cuda")
    # multi(model_path, image_dir, meta_path, save_dir, mode="cuda")
