import onnx
import onnxruntime as ort
import numpy as np
import time

from read_utils import *


def get_onnx_model(onnx_path: str, use_cuda: bool=False) -> ort.InferenceSession:
    """获取onnxruntime模型

    Args:
        onnx_path (str): 模型路径
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.

    Returns:
        ort.InferenceSession: 模型session
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if use_cuda:
        model = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider"], provider_options=[{"device_id": 0}])
    else:
        model = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    return model


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
    onnx_model = get_onnx_model(model_path, use_cuda)

    # 2.打开图片
    image, origin_height, origin_width = load_image(image_path)

    # 3.获取meta_data
    meta_data = get_meta_data(param_dir)
    # 推理时使用的图片大小
    pred_image_height, pred_image_width = meta_data["img_size"]
    meta_data["image_shape"] = [origin_height, origin_width]

    start = time.time()
    # 4.图片预处理
    transform = get_transform(pred_image_height, pred_image_width, tensor=False)
    x = transform(image=image)
    x = np.expand_dims(x['image'], axis=0)
    # x = np.ones((1, 3, 224, 224))
    x = x.astype(dtype=np.float32)

    # 5.预测得到热力图和概率
    inputs = onnx_model.get_inputs()
    input_name1 = inputs[0].name
    results = onnx_model.run(None, {input_name1: x})
    anomaly_map, pred_score = results
    print("pred_score:", pred_score)    # 3.0487

    # 6.后处理,归一化热力图和概率,保存图片
    output, pred_score = post(anomaly_map, pred_score, image, meta_data)
    end = time.time()

    print("pred_score:", pred_score)    # 0.8933535814285278
    print("infer time:", end - start)
    cv2.imwrite(save_img_dir, output)



if __name__ == "__main__":
    image_path   = "./datasets/MVTec/bottle/test/broken_large/000.png"
    model_path   = "./results/patchcore/mvtec/bottle-cls/optimization/model.onnx"
    param_dir    = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_img_dir = "./results/patchcore/mvtec/bottle-cls/onnx_output.jpg"
    predict(model_path, image_path, param_dir, save_img_dir, use_cuda=False)
