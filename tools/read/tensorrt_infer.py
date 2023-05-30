import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 初始化cuda
import time
import os
from statistics import mean
import numpy as np
from infer import Inference
from read_utils import *


# refer https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/infer.py
class TrtInference(Inference):
    def __init__(self, model_path: str, *args, **kwargs) -> None:
        """
        Args:
            model_path (str):   model_path
        """
        super().__init__(*args, **kwargs)
        # 1.载入模型
        self.get_model(model_path)
        # 2.预热模型
        # self.warm_up()


    def get_model(self, engine_path: str):
        """获取tensorrt模型

        Args:
            engine_path (str):  模型路径

        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []        # inputs
        self.outputs = []       # outputs
        self.allocations = []   # inputs&outputs cuda memorys
        for i in range(self.engine.num_bindings):
            is_input = False
            if trt.__version__ < "8.5":
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
            else:
                # 8.5 api
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name) # both engine and context have this func and the returns are same.

            if is_input:
                self.batch_size = shape[0]
            dtype = np.dtype(trt.nptype(dtype))
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)                               # allocate cuda memory
            host_allocation = None if is_input else np.zeros(shape, dtype)  # allocate memory
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation) # allocate cuda memory
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self, i:int = 0):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[i]["shape"], self.inputs[i]["dtype"]

    def output_spec(self, i:int = 0):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[i]["shape"], self.outputs[i]["dtype"]

    def infer(self, image: np.ndarray) -> tuple[np.ndarray]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray]: anomaly_map, score
        """

        # 1.推理
        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(image))             # cpu memory to gpu memory
        self.context.execute_v2(self.allocations)

        if len(self.outputs) == 1:
            cuda.memcpy_dtoh(self.outputs[0]["host_allocation"], self.outputs[0]["allocation"]) # gpu memory to cpu memory
            anomaly_map = np.reshape(self.outputs[0]["host_allocation"], (1, 1, self.infer_height, self.infer_width))
            pred_score  = anomaly_map.reshape(-1).max() # [1]
        else:
            # patchcore有两个输出结果 [1]代表anomaly_map [0]代表pred_score
            cuda.memcpy_dtoh(self.outputs[1]["host_allocation"], self.outputs[1]["allocation"]) # gpu memory to cpu memory
            anomaly_map = self.outputs[1]["host_allocation"]
            cuda.memcpy_dtoh(self.outputs[0]["host_allocation"], self.outputs[0]["allocation"])
            pred_score = self.outputs[0]["host_allocation"]

        return anomaly_map, pred_score


if __name__ == "__main__":
    # 注意使用非patchcore模型时报错可以查看infer中infer_height和infer_width中的[1] 都改为 [0]，具体查看注释和metadata.json文件
    image_path = "../../datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "../../datasets/MVTec/bottle/test/broken_large"
    model_path = "../../results/patchcore/mvtec/bottle/run/weights/openvino/model.engine"
    meta_path  = "../../results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json"
    save_path  = "../../results/patchcore/mvtec/bottle/run/weights/openvino/result.jpg"
    save_dir   = "../../results/patchcore/mvtec/bottle/run/weights/openvino/result"
    infer = TrtInference(model_path=model_path, meta_path=meta_path)
    infer.single(image_path=image_path, save_path=save_path)
    # infer.multi(image_dir=image_dir, save_dir=save_dir)
