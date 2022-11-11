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

from infer import Inference
from read_utils import *


"""openvino图片预处理方法

# input().tensor()       有7个方法
ppp.input().tensor().set_color_format().set_element_type().set_layout() \
                    .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape()

# output().tensor()      有2个方法
ppp.output().tensor().set_layout().set_element_type()

# input().preprocess()   有8个方法
ppp.input().preprocess().convert_color().convert_element_type().mean().scale() \
                        .convert_layout().reverse_channels().resize().custom()

# output().postprocess() 有3个方法
ppp.output().postprocess().convert_element_type().convert_layout().custom()

# input().model()  只有1个方法
ppp.input().model().set_layout()

# output().model() 只有1个方法
ppp.output().model().set_layout()
"""


class OVInference(Inference):
    def __init__(self, model_path: str, meta_path: str, mode: str = 'CPU', openvino_preprocess: bool = False) -> None:
        """
        Args:
            model_path (str): 模型路径
            meta_path (str): 超参数路径
            mode (str, optional): CPU or GPU or GPU.0  Defaults to CPU. 具体可以使用设备可以运行 samples/python/hello_query_device/hello_query_device.py 文件查看
            openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
        """
        super().__init__()
        self.openvino_preprocess = openvino_preprocess
        # 超参数
        self.meta  = get_meta_data(meta_path)
        # 载入模型
        self.model = self.get_openvino_model(model_path, mode)
        # 预热模型
        self.warm_up()


    def get_openvino_model(self, model_path: str, mode: str='CPU') -> ov.CompiledModel:
        """获取模型

        Args:
            model_path (str):       模路径, xml or onnx
            mode (str, optional): CPU or GPU. Defaults to CPU.

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
        if self.openvino_preprocess:
            # Step 4. Inizialize Preprocessing for the model  openvino数据预处理
            # https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
            # https://blog.csdn.net/sandmangu/article/details/107181289
            # https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
            ppp = PrePostProcessor(model)
            # 设定图片数据类型，形状，通道排布为RGB     input(0) 指的是第0个输入
            ppp.input(0).tensor().set_color_format(ColorFormat.RGB) \
                .set_element_type(Type.f32).set_layout(Layout("NCHW"))   # BGR -> RGB Type.u8 -> Type.f32  NHWC -> NCHW
            # 预处理: 改变类型,转换为RGB,减去均值,除以标准差(均值和标准差包含了归一化)
            # ppp.input(0).preprocess().convert_color(ColorFormat.RGB).convert_element_type(Type.f32).mean(mean).scale(std)
            ppp.input(0).preprocess().mean(mean).scale(std)
            # 指定模型输入形状
            ppp.input(0).model().set_layout(Layout("NCHW"))
            # 指定模型输出类型
            ppp.output(0).tensor().set_element_type(Type.f32)
            ppp.output(1).tensor().set_element_type(Type.f32)
            # Embed above steps in the graph
            model = ppp.build()

        compiled_model = core.compile_model(model, device_name=mode)
        return compiled_model


    def warm_up(self):
        """预热模型
        """
        # 预热模型
        infer_height, infer_width = self.meta["infer_size"]
        x = np.zeros((1, 3, infer_height, infer_width), dtype=np.float32)
        self.model([x])


    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: hotmap, score
        """
        # 1.保存原图宽高
        self.meta["image_size"] = [image.shape[0], image.shape[1]]

        # 1.获取模型输入,输出
        inputs  = self.model.inputs
        outputs = self.model.outputs
        # print(f"inputs: {inputs}")      # inputs: [<ConstOutput: names[input] shape{1,3,224,224} type: f32>]
        # print(f"outputs: {outputs}")    # outputs: [<ConstOutput: names[output] shape{1,1,224,224} type: f32>, <ConstOutput: names[278] shape{1} type: f32>]
        # 创建推理请求
        # infer_request = compiled_model.create_infer_request()

        # 2.图片预处理
        # 推理时使用的图片大小
        infer_height, infer_width = self.meta["infer_size"]
        if self.openvino_preprocess:
            # 使用openvino数据预处理要缩放图片
            x = cv2.resize(image, (infer_height, infer_width))
            x = x.transpose(2, 0, 1)            # [h, w, c] -> [c, h, w]
        else:
            transform = get_transform(infer_height, infer_width, tensor=False)
            x = transform(image=image)['image'] # [c, h, w]
        # [c, h, w] -> [b, c, h, w]
        x = np.expand_dims(x, axis=0)
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 3.预测得到热力图和概率
        # 推理 多种方式
        # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
        # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
        # results = infer_request.infer({inputs[0]: x})     # 同样支持list输入
        # results = compiled_model({inputs[0]: x})
        results = self.model([x])
        anomaly_map = results[outputs[0]]
        pred_score  = results[outputs[1]]
        print("pred_score:", pred_score)    # 3.1183267

        # 4.后处理,归一化热力图和概率
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

        return anomaly_map, pred_score


def single(model_path: str, image_path: str, meta_path: str,
            save_path: str, mode: str = 'CPU', openvino_preprocess: bool = False) -> None:
    """预测单张图片

    Args:
        model_path (str):   模型路径
        image_path (str):   图片路径
        meta_path (str):    超参数路径
        save_path (str):    保存图片路径
        mode (str, optional): CPU or GPU. Defaults to CPU.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
    """
    # 1.获取推理器
    inference = OVInference(model_path, meta_path, mode, openvino_preprocess)

    # 2.打开图片
    image = load_image(image_path)

    # 3.推理
    start = time.time()
    anomaly_map, pred_score = inference.infer(image)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885372877120972
    print("infer time:", end - start)

    # 4.保存图片
    save_image(save_path, image, anomaly_map, pred_score)


def multi(model_path: str, image_dir: str, meta_path: str,
            save_dir: str, mode: str = 'CPU', openvino_preprocess: bool = False) -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        image_dir (str):    图片文件夹
        meta_path (str):    超参数路径
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        mode (str, optional): CPU or GPU. Defaults to CPU.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.获取推理器
    inference = OVInference(model_path, meta_path, mode, openvino_preprocess)

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
        anomaly_map, pred_score = inference.infer(image)
        end = time.time()

        infer_times.append(end - start)
        scores.append(pred_score)
        print("pred_score:", pred_score)    # 0.8885372877120972
        print("infer time:", end - start)

        # 6.保存图片
        if save_dir is not None:
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, anomaly_map, pred_score)

    print("avg infer time: ", mean(infer_times))
    draw_score(scores, save_dir)


if __name__ == '__main__':
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle/run/optimization/openvino/model.xml"
    meta_path  = "./results/patchcore/mvtec/bottle/run/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle/run/openvino_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle/run/result"
    single(model_path, image_path, meta_path, save_path, mode='CPU', openvino_preprocess=True)
    # multi(model_path, image_dir, meta_path, save_dir, mode='CPU', openvino_preprocess=True)
