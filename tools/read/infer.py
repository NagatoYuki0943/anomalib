from abc import ABC, abstractmethod
import torch
import time
import cv2
import os
from statistics import mean

from read_utils import *


class Inference(ABC):

    def post(self, anomaly_map: Union[Tensor, np.ndarray],
            pred_score: Union[Tensor, np.float32],
            image: np.ndarray) -> Union[np.ndarray, float]:
        """后处理

        Args:
            anomaly_map (Union[Tensor, np.ndarray]): 热力图
            pred_score (Union[Tensor, np.float32]): 预测分数
            image (np.ndarray): 原图

        Returns:
            np.ndarray: 最终图片
            float: 最终得分
        """
        # 6.预测结果后处理，包括归一化热力图和概率，所放到原图尺寸
        anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)
        # print(anomaly_map.shape)            # (900, 900)
        # print("pred_score:", pred_score)    # 0.8933535814285278

        # 7.混合原图
        superimposed_map = superimpose_anomaly_map(anomaly_map, image)
        # print(superimposed_map.shape)                      # (900, 900, 3)

        # 8.添加标签
        output = add_label(superimposed_map, pred_score)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output, pred_score


    @abstractmethod
    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError
