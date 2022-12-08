import onnx
import onnxruntime
import numpy as np
import time

sess_options = onnxruntime.SessionOptions()
sess_options.register_custom_ops_library("outputs/custom_vortex_fp16.so")
session = onnxruntime.InferenceSession(
    "outputs/vortex_fp16.onnx",
    providers=["CPUExecutionProvider"],
    provider_options=[{}],
    sess_options=sess_options,
)

input_dict = {
    "q_title_token_ids": np.random.randint(256, size=[1, 512]).astype("int32"),
    "q_title_token_types": np.random.randint(256, size=[1, 512]).astype("int32"),
    "q_title_token_masks": np.random.randint(256, size=[1, 512]).astype("int32"),
}

tvm_output = session.run(output_names=None, input_feed=input_dict)

num_iters = 1000
start = time.time()
for i in range(num_iters):
    tvm_output = session.run(output_names=None, input_feed=input_dict)
end = time.time()

print("Runtime: %f ms" % ((1000 * (end - start)) / num_iters))
