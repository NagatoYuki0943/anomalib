import onnx
import onnxruntime as ort
import numpy as np


onnx_path = "./results/patchcore/mvtec/bottle-cls/optimization/model.onnx"


so = ort.SessionOptions()
so.log_severity_level = 3

net = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
# net = ort.InferenceSession(onnx_path, so, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])

x = np.ones((1, 3, 224, 224))
x = x.astype(dtype=np.float32)

input_name = net.get_inputs()[0].name
output_name = net.get_outputs()[0].name


out = net.run(None, {input_name: x})
print(out[0].shape)
print(out[1])

