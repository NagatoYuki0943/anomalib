import onnx
import onnxruntime as ort
import numpy as np


onnx_path = "./results/output_512_0.01.onnx"


so = ort.SessionOptions()
so.log_severity_level = 3

net = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
# net = ort.InferenceSession(onnx_path, so, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])

x = np.random.randn(1, 3, 512, 512)
x = x.astype(dtype=np.float32)
print(x.shape)      # (1, 3, 512, 512)

input_name = net.get_inputs()[0].name
output_name = net.get_outputs()[0].name

# print(input_name)   # input
# print(output_name)  # output


out = net.run(None, {input_name: x})
print(out)

