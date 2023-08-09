from abc import ABC, abstractmethod
import numpy as np
from read_utils import get_json, load_image, gen_images, save_image, draw_score, post_process, get_transform
import time
import os
from statistics import mean


class Inference(ABC):
    def __init__(self, meta_path: str, openvino_preprocess: bool = False, efficient_ad: bool = False) -> None:
        """
        Args:
            meta_path (str):            超参数路径
            openvino_preprocess (bool): 是否使用openvino图片预处理,只有openvino模型使用
            efficient_ad (bool): 是否使用efficient_ad模型
        """
        super().__init__()
        # 1.超参数
        self.meta  = get_json(meta_path)
        # 2.openvino图片预处理
        self.openvino_preprocess = openvino_preprocess
        self.efficient_ad = efficient_ad
        # 3.transform
        self.infer_height = self.meta["transform"]["transform"]["transforms"][0]["height"] # 推理时使用的图片大小
        self.infer_width  = self.meta["transform"]["transform"]["transforms"][0]["width"]
        if openvino_preprocess:
            self.transform = get_transform(self.infer_height, self.infer_width, self.efficient_ad, "openvino")
        else:
            self.transform = get_transform(self.infer_height, self.infer_width, self.efficient_ad, "numpy")

    @abstractmethod
    def infer(self, image: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError

    def warm_up(self):
        """预热模型
        """
        # [h w c], 这是opencv读取图片的shape
        x = np.zeros((1, 3, self.infer_height, self.infer_width), dtype=np.float32)
        self.infer(x)

    def single(self, image_path: str, save_path: str) -> None:
        """预测单张图片

        Args:
            image_path (str):   图片路径
            save_path (str):    保存图片路径
        """
        # 1.打开图片
        image = load_image(image_path)

        # 2.保存原图高宽
        self.meta["image_size"] = [image.shape[0], image.shape[1]]

        # 3.图片预处理
        x = self.transform(image=image)['image']    # [c, h, w]
        x = np.expand_dims(x, axis=0)               # [c, h, w] -> [b, c, h, w]
        x = x.astype(dtype=np.float32)

        # 4.推理
        start = time.time()
        anomaly_map, pred_score = self.infer(x)     # [900, 900] [1]

        # 5.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        print("pred_score:", pred_score)    # 0.8885370492935181
        print("infer time:", (end - start) * 1000, "ms")

        # 7.保存图片
        save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)

    def multi(self, image_dir: str, save_dir: str=None) -> None:
        """预测多张图片

        Args:
            image_dir (str):    图片文件夹
            save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        """
        # 0.检查保存路径
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"mkdir {save_dir}")
        else:
            print("保存路径为None,不会保存图片")

        # 1.获取文件夹中图片
        imgs = os.listdir(image_dir)
        imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

        infer_times: list[float] = []
        scores: list[float] = []
        # 批量推理
        for img in imgs:
            # 2.拼接图片路径
            image_path = os.path.join(image_dir, img);

            # 3.打开图片
            image = load_image(image_path)

            # 4.图片预处理
            x = self.transform(image=image)['image']    # [c, h, w]
            x = np.expand_dims(x, axis=0)               # [c, h, w] -> [b, c, h, w]
            x = x.astype(dtype=np.float32)

            # 5.推理
            start = time.time()
            anomaly_map, pred_score = self.infer(x)     # [900, 900] [1]

            # 6.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
            anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

            # 7.生成mask,mask边缘,热力图叠加原图
            mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
            end = time.time()

            infer_times.append(end - start)
            scores.append(pred_score)
            print("pred_score:", pred_score)    # 0.8885370492935181
            print("infer time:", (end - start) * 1000, "ms")

            if save_dir is not None:
                # 7.保存图片
                save_path = os.path.join(save_dir, img)
                save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)

        print("avg infer time: ", mean(infer_times) * 1000, "ms")
        draw_score(scores, save_dir)