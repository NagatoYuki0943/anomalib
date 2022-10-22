import numpy as np
import json
import cv2
import torch
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Tuple, Optional
from omegaconf import DictConfig
import time


#-----------------------------#
#   打开图片
#-----------------------------#
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                  # BGR2RGB
    origin_height = image.shape[0]
    origin_width  = image.shape[1]
    return image, origin_height, origin_width


#-----------------------------#
#   图片预处理
#-----------------------------#
def get_transform(height, width):
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 归一化+标准化
            ToTensorV2(),
        ]
    )


#-----------------------------#
#   获取meta_data
#-----------------------------#
def get_meta_data(jsonpath):
    """获取 image_threshold, pixel_threshold, min, max

    Args:
        jsonpath (str): json file path

    Returns:
        meta_data(dict): metadata
    """
    with open(jsonpath, mode='r', encoding='utf-8') as f:
        meta_data = json.load(f)
    # print(meta_data)
    return meta_data


#-----------------------------#
#   分别标准化热力图和得分
#   anomalib.post_processing.normalization.min_max.normalize
#-----------------------------#
def normalize_min_max(
    targets: Union[np.ndarray, Tensor, np.float32],
    threshold: Union[np.ndarray, Tensor, float],
    min_val: Union[np.ndarray, Tensor, float],
    max_val: Union[np.ndarray, Tensor, float],
) -> Union[np.ndarray, Tensor]:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized



#-----------------------------#
#   标准化热力图和得分
#-----------------------------#
def normalize(
    anomaly_maps: Union[Tensor, np.ndarray],
    pred_scores: Union[Tensor, np.float32],
    meta_data: Union[Dict, DictConfig],
) -> Tuple[Union[np.ndarray, Tensor], float]:
    """Applies normalization and resizes the image.

    Args:
        anomaly_maps (Union[Tensor, np.ndarray]): Predicted raw anomaly map.            torch.Size([224, 224])
        pred_scores (Union[Tensor, np.float32]): Predicted anomaly score                tensor(1.0392)
        meta_data (Dict): Meta data. Post-processing step sometimes requires
            additional meta data such as image shape. This variable comprises such info.

    Returns:
        Tuple[Union[np.ndarray, Tensor], float]: Post processed predictions that are ready to be visualized and
            predicted scores.
    """

    # min max normalization
    if "min" in meta_data and "max" in meta_data:
        # 热力图标准化
        anomaly_maps = normalize_min_max(
            anomaly_maps, meta_data["pixel_threshold"], meta_data["min"], meta_data["max"]
        )
        # tensor(1.0392) -> tensor(0.5510)
        pred_scores = normalize_min_max(
            pred_scores, meta_data["image_threshold"], meta_data["min"], meta_data["max"]
        )
    return anomaly_maps, float(pred_scores)


#-----------------------------#
#   预测结果后处理，包括归一化，所放到原图尺寸
#-----------------------------#
def post_process(
    anomaly_map: Tensor, pred_score: Tensor, meta_data: Optional[Union[Dict, DictConfig]] = None
) -> Tuple[np.ndarray, float]:
    """Post process the output predictions.
        这里返回的结果和inference的图像结果有些许不同,小数点后几位不同,不过得分相同
    Args:
        predictions (Tensor): Raw output predicted by the model.
        meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
            additional meta data such as image shape. This variable comprises such info.
            Defaults to None.

    Returns:
        tuple(np.ndarray, float):还原到原图尺寸的热力图和得分
    """
    anomaly_map = anomaly_map.squeeze()             # [1, 1, 224, 224] -> [224, 224]

    #------------------------------#
    #   标准化
    #------------------------------#
    anomaly_map, pred_score = normalize(anomaly_map, pred_score, meta_data)

    if isinstance(anomaly_map, Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    #------------------------------#
    #   所放到原图尺寸
    #------------------------------#
    if "image_shape" in meta_data and anomaly_map.shape != meta_data["image_shape"]:
        image_height = meta_data["image_shape"][0]
        image_width = meta_data["image_shape"][1]
        anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

    # 返回图像和得分给inference.py
    return anomaly_map, float(pred_score)


#-----------------------------#
#   单通道热力图转换为rgb
#-----------------------------#
def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """ 单通道热力图转换为rgb
        Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.        [900, 900]
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]                                                           [900, 900, 3]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)   # np.ptp()函数实现的功能等同于np.max(array) - np.min(array)
    anomaly_map = anomaly_map * 255                                             # 0~1 -> 0~255
    anomaly_map = anomaly_map.astype(np.uint8)                                  # 变为整数

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)              # [900, 900] -> [900, 900, 3]
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_RGB2BGR)                  # RGB2BGR
    return anomaly_map


#-----------------------------#
#   将热力图和原图叠加
#-----------------------------#
def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """将热力图和原图叠加
        Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map       热力图  [900, 900]
        image (np.ndarray): Input image             原图    [900, 900]
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max

    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    # 单通道热力图转换为rgb [900, 900] -> [900, 900, 3]
    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    # 叠加图片
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


#-----------------------------#
#   给图片添加概率
#-----------------------------#
def add_label(prediction: np.ndarray, scores: float, font: int = cv2.FONT_HERSHEY_PLAIN) -> np.ndarray:
    """If the model outputs score, it adds the score to the output image.

    Args:
        prediction (np.ndarray): Resized anomaly map.
        scores (float): Confidence score.

    Returns:
        np.ndarray: Image with score text.
    """
    text = f"Confidence Score {scores:.0%}"
    font_size = prediction.shape[1] // 1024 + 1  # Text scale is calculated based on the reference size of 1024
    (width, height), baseline = cv2.getTextSize(text, font, font_size, thickness=font_size // 2)
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = (225, 252, 134)
    cv2.putText(label_patch, text, (0, baseline // 2 + height), font, font_size, 0, lineType=cv2.LINE_AA)
    prediction[: baseline + height, : baseline + width] = label_patch
    return prediction


#-----------------------------#
#   预测函数
#-----------------------------#
def predict(image_path: str, torchscript_path: str, param_dir: str, save_img_dir: str, use_cuda: bool=False) -> None:

    # 1.打开图片
    image, origin_height, origin_width = load_image(image_path)
    # 2.获取meta_data
    meta_data = get_meta_data(param_dir)
    # 推理时使用的图片大小
    pred_image_height, pred_image_width = meta_data["img_size"]
    meta_data["image_shape"] = [origin_height, origin_width]

    # 3.读取模型
    trace_model = torch.jit.load(torchscript_path)

    start = time.time()
    # 4.图片预处理
    transform = get_transform(pred_image_height, pred_image_width)
    image_tensor = transform(image=image)
    image_tensor = image_tensor['image'].unsqueeze(0)
    if use_cuda:
        image_tensor = image_tensor.cuda()

    # 5.预测得到热力图和概率
    anomaly_map, pred_score = trace_model(image_tensor)
    print("pred_score:", pred_score)    # 3.0487

    # 6.预测结果后处理，包括归一化热力图和概率，所放到原图尺寸
    anomaly_map, pred_score = post_process(anomaly_map, pred_score, meta_data)
    # print(anomaly_map.shape)            # (900, 900)
    print("pred_score:", pred_score)    # 0.8933535814285278

    # 7.混合原图
    superimposed_map = superimpose_anomaly_map(anomaly_map, image)
    # print(superimposed_map.shape)                      # (900, 900, 3)

    # 8.添加标签
    output = add_label(superimposed_map, pred_score)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    end = time.time()
    print("infer time:", end-start) # infer time: 2.19

    # 9.写入图片
    cv2.imwrite(filename=save_img_dir, img=output)


if __name__ == "__main__":
    image_path       = "./datasets/MVTec/bottle/test/broken_large/000.png"
    torchscript_path = "./results/patchcore/mvtec/bottle-cls/optimization/model_cpu.torchscript"
    param_dir        = "./results/patchcore/mvtec/bottle-cls/optimization/meta_data.json"
    save_img_dir     = "./results/patchcore/mvtec/bottle-cls/output.jpg"
    predict(image_path, torchscript_path, param_dir, save_img_dir, use_cuda=False)
