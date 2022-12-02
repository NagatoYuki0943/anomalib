import torch
from torch import Tensor
import numpy as np
import time
import os
from statistics import mean

from infer import Inference
from read_utils import load_image, get_transform, get_json, post_process, gen_images, save_image, draw_score


class TorchscriptInference(Inference):
    def __init__(self, model_path: str, meta_path: str, use_cuda: bool=False) -> None:
        """
        Args:
            model_path (str): 模型路径
            meta_path (str): 超参数路径
            use_cuda (bool, optional): 是否使用cuda. Defaults to False.
        """
        super().__init__()
        self.use_cuda = use_cuda
        # 超参数
        self.meta = get_json(meta_path)
        # 载入模型
        self.model = self.get_model(model_path)
        self.model.eval()
        # 预热模型
        self.warm_up()


    def get_model(self, torchscript_path: str):
        """获取script模型

        Args:
            torchscript_path (str): 模型路径

        Returns:
            torchscript: script模型
        """
        return torch.jit.load(torchscript_path)


    def warm_up(self):
        """预热模型
        """
        infer_height, infer_width = self.meta["infer_size"]
        x = torch.zeros(1, 3, infer_height, infer_width)
        if self.use_cuda:
            x = x.cuda()
        with torch.inference_mode():
            self.model(x)


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
        # 推理时使用的图片大小
        infer_height, infer_width = self.meta["infer_size"]
        transform = get_transform(infer_height, infer_width, tensor=True)
        x = transform(image=image)
        x = x['image'].unsqueeze(0)
        # x = torch.ones(1, 3, 224, 224)

        # 2.预测得到热力图和概率
        if self.use_cuda:
            x = x.cuda()
        with torch.inference_mode():
            predictions = self.model(x)     # 单个值返回Tensor,多个值返回tuple

        # 3.解决不同模型输出问题
        if isinstance(predictions, Tensor):
            # 大多数模型只返回热力图
            # https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/deploy/inferencers/torch_inferencer.py#L159
            anomaly_map = predictions.detach().cpu().numpy()    # [1, 1, 256, 256]
            pred_score  = anomaly_map.reshape(-1).max()         # [1]
        else:
            # patchcore返回热力图和得分
            anomaly_map, pred_score = predictions
            anomaly_map = anomaly_map.detach().cpu().numpy()
            pred_score  = pred_score.detach().cpu().numpy()
        print("pred_score:", pred_score)    # 3.1183

        # 4.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)

        return anomaly_map, pred_score


def single(model_path: str, meta_path: str, image_path: str, save_path: str, use_cuda: bool=False) -> None:
    """预测单张图片

    Args:
        model_path (str):   模型路径
        meta_path (str):    超参数路径
        image_path (str):   图片路径
        save_path (str):    保存图片路径
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 1.获取推理器
    inference = TorchscriptInference(model_path, meta_path, use_cuda)

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
    save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)


def multi(model_path: str, meta_path: str, image_dir: str, save_dir: str, use_cuda: bool=False) -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        meta_path (str):    超参数路径
        image_dir (str):    图片文件夹
        save_dir (str, optional):  保存图片路径,没有就不保存. Defaults to None.
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.获取推理器
    inference = TorchscriptInference(model_path, meta_path, use_cuda)

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

        if save_dir is not None:
            # 7.保存图片
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)

    print("avg infer time: ", mean(infer_times))
    draw_score(scores, save_dir)


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    model_path = "./results/patchcore/mvtec/bottle/256/optimization/model_gpu.torchscript"
    meta_path  = "./results/patchcore/mvtec/bottle/256/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle/256/torchscript_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle/256/result"
    single(model_path, meta_path, image_path, save_path, use_cuda=True)   # 注意: 使用cuda时要使用gpu模型
    # multi(model_path, meta_path, image_dir, save_dir, use_cuda=True)    # 注意: 使用cuda时要使用gpu模型
