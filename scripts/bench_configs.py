"""Test different TFLite configurations to find fastest invoke()."""
import time
import os
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

img = np.random.rand(1, 224, 224, 3).astype(np.float32)
model_path = "models/classifier.tflite"
cpu_count = os.cpu_count() or 4

configs = [
    ("default (num_threads=None)", {}),
    ("num_threads=1", {"num_threads": 1}),
    ("num_threads=2", {"num_threads": 2}),
    ("num_threads=4", {"num_threads": 4}),
    (f"num_threads={cpu_count}", {"num_threads": cpu_count}),
]

for label, kwargs in configs:
    interp = Interpreter(model_path=model_path, **kwargs)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    # warmup
    interp.set_tensor(inp[0]["index"], img)
    interp.invoke()

    # benchmark 3 calls
    times = []
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    print(f"{label:40s} -> mean: {mean:7.1f}ms  [{', '.join(f'{t:.0f}' for t in times)}]")
