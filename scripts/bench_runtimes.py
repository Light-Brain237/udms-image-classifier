"""Compare performance: ai_edge_litert vs tensorflow_cpu interpreter."""
import time
import numpy as np

model_path = "models/classifier.tflite"
img = np.random.rand(1, 224, 224, 3).astype(np.float32)

def bench(label, interp):
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]["index"], img)
    interp.invoke()  # warmup
    times = []
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    print(f"{label:50s} mean={mean:7.1f}ms  {[f'{t:.0f}' for t in times]}")

# 1. ai_edge_litert default
from ai_edge_litert.interpreter import Interpreter as AiInterp
bench("ai_edge_litert (default)", AiInterp(model_path=model_path))
bench("ai_edge_litert (threads=4)", AiInterp(model_path=model_path, num_threads=4))

# 2. tensorflow.lite
from tensorflow.lite.python.interpreter import Interpreter as TfInterp
bench("tf.lite (default)", TfInterp(model_path=model_path))
bench("tf.lite (threads=4)", TfInterp(model_path=model_path, num_threads=4))

# 3. tensorflow.lite with experimental_delegates
try:
    from tensorflow.lite.python.interpreter import load_delegate
    # On Windows, try XNNPACK via the built-in delegate
    # TFLite 2.x bundles XNNPACK; we just need experimental_op_resolver_type
    pass
except Exception as e:
    print(f"load_delegate: {e}")

# 4. Check if model has metadata about quantization
interp = TfInterp(model_path=model_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"\nModel details:")
print(f"  Input:  dtype={inp['dtype']}, shape={inp['shape']}, quantization={inp.get('quantization')}")
print(f"  Output: dtype={out['dtype']}, shape={out['shape']}, quantization={out.get('quantization')}")
