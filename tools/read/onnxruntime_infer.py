import onnxruntime as ort
import numpy as np
from infer import Inference

print(ort.__version__)
# print("onnxruntime all providers:", ort.get_all_providers())
print("onnxruntime available providers:", ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
print(ort.get_device())
# GPU


class OrtInference(Inference):
    def __init__(self, model_path: str, mode: str="cpu", *args, **kwargs) -> None:
        """
        Args:
            model_path (str): 模型路径
            meta_path (str): 超参数路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__(*args, **kwargs)
        # 1.载入模型
        self.model = self.get_model(model_path, mode)
        # 2.获取输入输出
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        # 3.预热模型
        self.warm_up()

    def get_model(self, onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
        """获取onnxruntime模型

        Args:
            onnx_path (str):    模型路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.

        Returns:
            ort.InferenceSession: 模型session
        """
        mode = mode.lower()
        assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."
        print(f"inference with {mode} !")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = {
            "cpu":  ['CPUExecutionProvider'],
            # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            "cuda": [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ],
            # tensorrt
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
            # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
            "tensorrt": [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 2GB
                        'trt_fp16_enable': False,
                    }),
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    })
                ]
        }[mode]
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    def infer(self, image: np.ndarray) -> tuple[np.ndarray]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: hotmap, score
        """
        # 1.推理
        predictions = self.model.run(None, {self.inputs[0].name: image})    # 返回值为list

        # 2.解决不同模型输出问题
        if len(predictions) != 2:
            # 大多数模型只返回热力图,efficient_ad有3个输出,不过也只需要第1个输出
            # https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/deploy/inferencers/torch_inferencer.py#L159
            anomaly_map = predictions[0]                # [1, 1, 256, 256] 返回类型为 np.ndarray
            pred_score  = anomaly_map.reshape(-1).max() # [1]
        else:
            # patchcore返回热力图和得分
            anomaly_map, pred_score = predictions
        print("pred_score:", pred_score)    # 3.1183257

        return anomaly_map, pred_score


if __name__ == "__main__":
    # patchcore模型训练配置文件删除了center_crop
    model_path = "../../results/efficient_ad/mvtec/bottle/run/weights/openvino/model.onnx"
    meta_path  = "../../results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json"
    image_path = "../../datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "../../datasets/MVTec/bottle/test/broken_large"
    save_path  = "../../results/efficient_ad/mvtec/bottle/run/weights/openvino/result.jpg"
    save_dir   = "../../results/efficient_ad/mvtec/bottle/run/weights/openvino/result"
    efficient_ad = True
    infer = OrtInference(model_path=model_path, meta_path=meta_path, mode="cuda", efficient_ad=efficient_ad)
    infer.single(image_path=image_path, save_path=save_path)
    # infer.multi(image_dir=image_dir, save_dir=save_dir)
