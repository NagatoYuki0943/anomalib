import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Union, Dict, Tuple
from omegaconf import DictConfig
from torch import Tensor
import numpy as np
import json
import cv2

script_path = "./results/output.torchscript"
image_path = "./datasets/some/1.abnormal/OriginImage_20220526_113038_Cam1_2_crop.jpg"


#-----------------------------#
# 分别标准化热力图和得分
#-----------------------------#
def normalize_min_max(
    targets: Union[np.ndarray, Tensor, np.float32],
    threshold: Union[np.ndarray, Tensor, float],
    min_val: Union[np.ndarray, Tensor, float],
    max_val: Union[np.ndarray, Tensor, float],
) -> Union[np.ndarray, Tensor]:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized


#-----------------------------#
# 标准化热力图和得分
#-----------------------------#
def normalize(
    anomaly_maps: Union[Tensor, np.ndarray],
    pred_scores: Union[Tensor, np.float32],
    meta_data: Union[Dict, DictConfig],
) -> Tuple[Union[np.ndarray, Tensor], float]:
    """Applies normalization and resizes the image.

    Args:
        anomaly_maps (Union[Tensor, np.ndarray]): Predicted raw anomaly map.            torch.Size([512, 512])
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
# 单通道热力图转换为rgb
#-----------------------------#
def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """ 单通道热力图转换为rgb
        Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.        [2711, 5351]
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]                                                           [2711, 5351, 3]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255 # 0~1 -> 0~255
    anomaly_map = anomaly_map.astype(np.uint8)  # 变为整数

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)  # [2711, 5351] -> [2711, 5351, 3]
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map


#-----------------------------#
# 将热力图和原图叠加
#-----------------------------#
def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """将热力图和原图叠加
        Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map       热力图  [2711, 5351]
        image (np.ndarray): Input image             原图    [2711, 5351]
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

    # 单通道热力图转换为rgb [2711, 5351] -> [2711, 5351, 3]
    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    # 叠加图片
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


#-----------------------------#
# 获取meta_data
#-----------------------------#
def get_meta_data(jsonpath):
    """获取 image_threshold, pixel_threshold, min, max

    Args:
        jsonpath (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(jsonpath, mode='r', encoding='utf-8') as f:
        meta_data = json.load(f)
    print(meta_data)
    return meta_data
meta_data = get_meta_data("./results/param.json")


#-----------------------------#
# 给图片添加概率
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



transform = A.Compose(
    [
        A.Resize(height=512, width=512, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


# 打开图片
image = Image.open(image_path)
origin_height = image.height
origin_width  = image.width
image = np.array(image)
image_tensor = transform(image=image)
image_tensor = image_tensor['image'].unsqueeze(0)

# 读取模型
trace_module = torch.jit.load(script_path)

# 得到结果
with torch.no_grad():
    anomaly_map, score = trace_module(image_tensor)

print(anomaly_map.size())                           # [1, 1, 512, 512]  没有经过标准化
print(score)                                        # tensor(1.2107)
anomaly_map = anomaly_map.squeeze()                 # [1, 1, 512, 512] -> [512, 512]
anomaly_map = anomaly_map.detach().numpy()
score = score.detach().numpy()

# 归一化数据
anomaly_map, score = normalize(anomaly_map, score, meta_data)
print(score)                                        # 0.6027931678867513

# 还原anomalib图片大小
print(anomaly_map.shape)                            # (512, 512)
anomaly_map = cv2.resize(anomaly_map, (origin_width, origin_height))
print(anomaly_map.shape)                            # (2711, 5351)

# 混合原图
superimposed_map = superimpose_anomaly_map(anomaly_map, image)
print(superimposed_map.shape)                       # (2711, 5351, 3)

# 添加标签
output = add_label(superimposed_map, score)

# 写入图片
cv2.imwrite(filename="./results/output.jpg", img=output)