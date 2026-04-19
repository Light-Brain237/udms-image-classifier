"""Benchmark invoke() directly to establish baseline timing."""
import time
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path="models/classifier.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

img = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Warm up
interpreter.set_tensor(inp[0]["index"], img)
interpreter.invoke()

# Benchmark
times = []
for _ in range(5):
    interpreter.set_tensor(inp[0]["index"], img)
    t0 = time.perf_counter()
    interpreter.invoke()
    times.append((time.perf_counter() - t0) * 1000)

print(f"invoke() times: {[f'{t:.1f}ms' for t in times]}")
print(f"mean: {sum(times)/len(times):.1f}ms")
