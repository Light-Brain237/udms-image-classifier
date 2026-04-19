"""Check TFLite runtime and test XNNPACK delegate."""
import sys
import time
import numpy as np

print(f"Python: {sys.version}")

try:
    from ai_edge_litert.interpreter import Interpreter
    import ai_edge_litert
    print(f"Runtime: ai_edge_litert {ai_edge_litert.__version__}")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    import tensorflow as tf
    print(f"Runtime: tensorflow {tf.__version__}")

# Try loading with XNNPACK delegate
model_path = "models/classifier.tflite"
img = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Method 1: Try experimental_delegates with load_delegate
try:
    try:
        from ai_edge_litert.interpreter import load_delegate
    except ImportError:
        from tensorflow.lite.python.interpreter import load_delegate
    xnnpack = load_delegate("libXNNPACK.so")
    interp = Interpreter(model_path=model_path, experimental_delegates=[xnnpack])
    print("XNNPACK loaded via load_delegate")
except Exception as e:
    print(f"load_delegate failed: {e}")

# Method 2: Try experimental_op_resolver_type
try:
    interp = Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=2,  # BUILTIN_WITHOUT_DEFAULT_DELEGATES = 0, BUILTIN_REF = 1, BUILTIN = 2
        num_threads=4,
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()
    interp.set_tensor(inp[0]["index"], img)
    # warmup
    interp.invoke()
    # bench
    times = []
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"op_resolver_type=2, threads=4: mean={sum(times)/len(times):.1f}ms  {[f'{t:.0f}' for t in times]}")
except Exception as e:
    print(f"op_resolver_type=2 failed: {e}")

# Method 3: Check tflite-runtime
try:
    import tflite_runtime
    print(f"tflite_runtime: {tflite_runtime.__version__}")
except ImportError:
    print("tflite_runtime: not installed")

# Check installed packages
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for line in result.stdout.splitlines():
    low = line.lower()
    if any(k in low for k in ["tflite", "tensorflow", "ai-edge", "litert"]):
        print(f"  Package: {line.strip()}")
