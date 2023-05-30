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
from read_utils import load_image, get_transform, get_json, post_process, gen_images, save_image, draw_score


"""openvino图片预处理方法
input(0)/output(0) 按照id找指定的输入输出,不指定找全部的输入输出

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
    def __init__(self, model_path: str, mode: str = 'CPU', *args, **kwargs) -> None:
        """
        Args:
            model_path (str): 模型路径
            mode (str, optional): CPU or GPU or GPU.0  Defaults to CPU. 具体可以使用设备可以运行 samples/python/hello_query_device/hello_query_device.py 文件查看
            openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        # 1.载入模型
        self.model = self.get_model(model_path, mode)
        # 2.保存模型输入输出
        self.inputs  = self.model.inputs
        self.outputs = self.model.outputs
        # 3.预热模型
        self.warm_up()


    def get_model(self, model_path: str, mode: str='CPU') -> ov.CompiledModel:
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
            # 设定图片数据类型，形状，通道排布为RGB
            ppp.input(0).tensor().set_color_format(ColorFormat.RGB) \
                .set_element_type(Type.f32).set_layout(Layout("NCHW"))   # BGR -> RGB Type.u8 -> Type.f32  NHWC -> NCHW
            # 预处理: 改变类型,转换为RGB,减去均值,除以标准差(均值和标准差包含了归一化)
            # ppp.input(0).preprocess().convert_color(ColorFormat.RGB).convert_element_type(Type.f32).mean(mean).scale(std)
            ppp.input(0).preprocess().mean(mean).scale(std)
            # 指定模型输入形状
            ppp.input(0).model().set_layout(Layout("NCHW"))
            # 指定模型输出类型
            ppp.output(0).tensor().set_element_type(Type.f32)
            if len(model.inputs) == 2:
                ppp.output(1).tensor().set_element_type(Type.f32)

            # Embed above steps in the graph
            model = ppp.build()

        compiled_model = core.compile_model(model, device_name=mode)
        return compiled_model

    def infer(self, image: np.ndarray) -> tuple[np.ndarray]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: anomaly_map, score
        """
        # 1.推理 多种方式
        # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
        # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#

        # 1.1 使用推理请求
        # infer_request = self.model.create_infer_request()
        # results       = infer_request.infer({self.inputs[0]: x})          # 直接返回推理结果
        # results       = infer_request.infer({0: x})                       # 直接返回推理结果
        # results       = infer_request.infer([x])                          # 直接返回推理结果
        # result0       = infer_request.get_output_tensor(outputs[0].index) # 通过方法获取单独结果  outputs[0].index 可以用0 1代替

        # 1.2 模型直接推理
        # results = self.model({self.inputs[0]: x})
        # results = self.model({0: x})
        results = self.model([image])

        # 2.解决不同模型输出问题 得到热力图和概率
        if len(self.outputs) == 1:
            # 大多数模型只返回热力图
            # https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/deploy/inferencers/torch_inferencer.py#L159
            anomaly_map = results[self.outputs[0]]          # [1, 1, 256, 256] 返回类型为 np.ndarray
            pred_score  = anomaly_map.reshape(-1).max()     # [1]
        else:
            # patchcore返回热力图和得分
            anomaly_map = results[self.outputs[0]]
            pred_score  = results[self.outputs[1]]
        print("pred_score:", pred_score)    # 3.1183267

        return anomaly_map, pred_score


if __name__ == '__main__':
    # 注意使用非patchcore模型时报错可以查看infer中infer_height和infer_width中的[1] 都改为 [0]，具体查看注释和metadata.json文件
    image_path = "../../datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "../../datasets/MVTec/bottle/test/broken_large"
    model_path = "../../results/patchcore/mvtec/bottle/run/weights/openvino/model.xml"
    meta_path  = "../../results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json"
    save_path  = "../../results/patchcore/mvtec/bottle/run/weights/openvino/result.jpg"
    save_dir   = "../../results/patchcore/mvtec/bottle/run/weights/openvino/result"
    infer = OVInference(model_path=model_path, meta_path=meta_path, mode="CPU", openvino_preprocess=True)
    infer.single(image_path=image_path, save_path=save_path)
    # infer.multi(image_dir=image_dir, save_dir=save_dir)
