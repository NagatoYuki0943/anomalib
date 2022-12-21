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
        # 1.超参数
        self.meta  = get_json(meta_path)
        # 2.载入模型
        self.model = self.get_model(model_path, mode)
        # 3.保存模型输入输出
        self.inputs  = self.model.inputs
        self.outputs = self.model.outputs
        # print(f"inputs: {self.inputs}")   # inputs: [<ConstOutput: names[input] shape{1,3,224,224} type: f32>]
        # print(f"outputs: {self.outputs}") # outputs: [<ConstOutput: names[output] shape{1,1,224,224} type: f32>, <ConstOutput: names[278] shape{1} type: f32>]
        # 4.transform
        infer_height, infer_width = self.meta["infer_size"] # 推理时使用的图片大小
        if self.openvino_preprocess:
            self.transform = get_transform(infer_height, infer_width, "openvino")
        else:
            self.transform = get_transform(infer_height, infer_width, "numpy")
        # 5.预热模型
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


    def warm_up(self):
        """预热模型
        """
        infer_height, infer_width = self.meta["infer_size"]
        # [h w c], 这是opencv读取图片的shape
        x = np.zeros((infer_height, infer_width, 3), dtype=np.float32)
        self.infer(x)


    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: hotmap, score
        """
        # 1.保存原图高宽
        self.meta["image_size"] = [image.shape[0], image.shape[1]]

        # 2.图片预处理
        x = self.transform(image=image)['image'] # [c, h, w]
        x = np.expand_dims(x, axis=0)            # [c, h, w] -> [b, c, h, w]
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 3.推理 多种方式
        # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
        # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#

        # 3.1 使用推理请求
        # infer_request = self.model.create_infer_request()
        # results       = infer_request.infer({self.inputs[0]: x})          # 直接返回推理结果
        # results       = infer_request.infer({0: x})                       # 直接返回推理结果
        # results       = infer_request.infer([x])                          # 直接返回推理结果
        # result0       = infer_request.get_output_tensor(outputs[0].index) # 通过方法获取单独结果  outputs[0].index 可以用0 1代替

        # 3.2 模型直接推理
        # results = self.model({self.inputs[0]: x})
        # results = self.model({0: x})
        results = self.model([x])

        # 4.解决不同模型输出问题 得到热力图和概率
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

        # 5.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

        return anomaly_map, pred_score


def single(model_path: str, meta_path: str, image_path: str,
            save_path: str, mode: str = 'CPU', openvino_preprocess: bool = False) -> None:
    """预测单张图片

    Args:
        model_path (str):   模型路径
        meta_path (str):    超参数路径
        image_path (str):   图片路径
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
    anomaly_map, pred_score = inference.infer(image)    # [900, 900] [1]

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885372877120972
    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)


def multi(model_path: str, meta_path: str, image_dir: str,
            save_dir: str, mode: str = 'CPU', openvino_preprocess: bool = False) -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        meta_path (str):    超参数路径
        image_dir (str):    图片文件夹
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        mode (str, optional): CPU or GPU. Defaults to CPU.
        openvino_preprocess (bool, optional): 是否使用openvino数据预处理. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
        anomaly_map, pred_score = inference.infer(image)    # [900, 900] [1]

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        infer_times.append(end - start)
        scores.append(pred_score)
        print("pred_score:", pred_score)    # 0.8885372877120972
        print("infer time:", end - start)

        if save_dir is not None:
            # 7.保存图片
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)

    print("avg infer time: ", mean(infer_times))
    draw_score(scores, save_dir)


if __name__ == '__main__':
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle/256/optimization/openvino/model.xml"
    meta_path  = "./results/patchcore/mvtec/bottle/256/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle/256/openvino_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle/256/openvino_result"
    single(model_path, meta_path, image_path, save_path, mode='CPU', openvino_preprocess=True)
    # multi(model_path, meta_path, image_dir, save_dir, mode='CPU', openvino_preprocess=True)
