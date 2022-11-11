from abc import ABC, abstractmethod
import cv2

from read_utils import *


class Inference(ABC):

    @abstractmethod
    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError
