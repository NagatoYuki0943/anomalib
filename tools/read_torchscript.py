import torch
import time
import cv2

from read_utils import *


def get_script_model(torchscript_path: str):
    return torch.jit.load(torchscript_path)


def predict(model_path: str, image_path: str, param_dir: str, save_img_dir: str, use_cuda: bool=False) -> None:
    """预测单张图片

    Args:
        model_path (str):   模型路径
        image_path (str):   图片路径
        param_dir (str):    超参数路径
        save_img_dir (str): 保存图片路径
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 1.读取模型
    trace_model = get_script_model(model_path)

    # 2.打开图片
    image, origin_height, origin_width = load_image(image_path)

    # 3.获取meta_data
    meta_data = get_meta_data(param_dir)
    # 推理时使用的图片大小
    infer_height, infer_width = meta_data["infer_size"]
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
    print("pred_score:", pred_score)    # 3.0487

    # 6.后处理,归一化热力图和概率,保存图片
    output, pred_score = post(anomaly_map, pred_score, image, meta_data)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8933535814285278
    print("infer time:", end - start)
    cv2.imwrite(save_img_dir, output)


if __name__ == "__main__":
    image_path   = "./datasets/MVTec/bottle/test/broken_large/000.png"
    model_path   = "./results/patchcore/mvtec/bottle-cls/optimization/model_cpu.torchscript"
    param_dir    = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_img_dir = "./results/patchcore/mvtec/bottle-cls/torchscript_output.jpg"
    predict(model_path, image_path, param_dir, save_img_dir, use_cuda=False)   # 注意: 使用cuda时要使用gpu模型
