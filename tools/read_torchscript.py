import torch
import time
import cv2
import os
from statistics import mean

from read_utils import *


def get_script_model(torchscript_path: str):
    return torch.jit.load(torchscript_path)


def single(model_path: str, image_path: str, param_dir: str, save_path: str, use_cuda: bool=False) -> None:
    """预测单张图片

    Args:
        model_path (str):    模型路径
        image_path (str):    图片路径
        param_dir (str):     超参数路径
        save_path (str): 保存图片路径
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 1.读取模型
    trace_model = get_script_model(model_path)

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
    transform = get_transform(infer_height, infer_width, tensor=True)
    x = transform(image=image)
    x = x['image'].unsqueeze(0)
    # x = torch.ones(1, 3, 224, 224)

    # 5.预测得到热力图和概率
    if use_cuda:
        x = x.cuda()
    with torch.inference_mode():
        anomaly_map, pred_score = trace_model(x)
    print("pred_score:", pred_score)    # 3.1183

    # 6.后处理,归一化热力图和概率,保存图片
    output, pred_score = post(anomaly_map, pred_score, image, meta_data)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8885370492935181
    print("infer time:", end - start)
    cv2.imwrite(save_path, output)


def multi(model_path: str, image_dir: str, param_dir: str, save_dir: str, use_cuda: bool=False) -> None:
    """预测多张图片

    Args:
        model_path (str):   模型路径
        image_dir (str):    图片文件夹
        param_dir (str):    超参数路径
        save_dir (str, optional):  保存图片路径,没有就不保存. Defaults to None.
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.读取模型
    trace_model = get_script_model(model_path)

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

        # 3.打开图片
        image, origin_height, origin_width = load_image(image_path)
        # 推理时使用的图片大小
        infer_height, infer_width = meta_data["infer_size"]
        # 保存原图宽高
        meta_data["image_size"] = [origin_height, origin_width]

        start = time.time()
        # 4.图片预处理
        transform = get_transform(infer_height, infer_width, tensor=True)
        x = transform(image=image)
        x = x['image'].unsqueeze(0)
        # x = torch.ones(1, 3, 224, 224)

        # 5.预测得到热力图和概率
        if use_cuda:
            x = x.cuda()
        with torch.inference_mode():
            anomaly_map, pred_score = trace_model(x)
        print("pred_score:", pred_score)    # 3.1183

        # 6.后处理,归一化热力图和概率,保存图片
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
    model_path = "./results/patchcore/mvtec/bottle-cls/optimization/model_cpu.torchscript"
    param_dir  = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_path  = "./results/patchcore/mvtec/bottle-cls/torchscript_output.jpg"
    save_dir   = "./results/patchcore/mvtec/bottle-cls/result"
    single(model_path, image_path, param_dir, save_path, use_cuda=False)   # 注意: 使用cuda时要使用gpu模型
    # multi(model_path, image_dir, param_dir, save_dir, use_cuda=False)   # 注意: 使用cuda时要使用gpu模型
