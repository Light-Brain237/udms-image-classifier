"""Investigate ai_edge_litert XNNPACK delegate capabilities."""
import sys
import os
import time
import numpy as np

# Find ai_edge_litert package location and check for delegate files
import ai_edge_litert
pkg_dir = os.path.dirname(ai_edge_litert.__file__)
print(f"ai_edge_litert location: {pkg_dir}")
print(f"ai_edge_litert version:  {ai_edge_litert.__version__}")

# List all files in the package
print("\nFiles in ai_edge_litert package:")
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), pkg_dir)
        if any(k in f.lower() for k in ['xnn', 'delegate', '.dll', '.pyd', '.so']):
            print(f"  ** {rel}")
        elif f.endswith(('.py', '.pyd', '.dll', '.so')):
            print(f"     {rel}")

# Check if ai_edge_litert.Interpreter supports experimental_delegates
from ai_edge_litert.interpreter import Interpreter
print(f"\nInterpreter init signature:")
import inspect
sig = inspect.signature(Interpreter.__init__)
print(f"  {sig}")

model_path = "models/classifier.tflite"
img = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Try 1: Check for load_delegate
print("\n--- Try load_delegate ---")
try:
    from ai_edge_litert.interpreter import load_delegate
    print(f"load_delegate available: {load_delegate}")
    # Check common Windows paths
    for name in ['XNNPACK.dll', 'libXNNPACK.dll', 'xnnpack.dll']:
        full = os.path.join(pkg_dir, name)
        print(f"  Checking {full}: exists={os.path.exists(full)}")
except ImportError as e:
    print(f"load_delegate not available: {e}")

# Try 2: experimental_delegates parameter
print("\n--- Try experimental_delegates ---")
try:
    # Some versions auto-create XNNPACK when you pass empty delegates list
    interp = Interpreter(model_path=model_path, experimental_delegates=None, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    interp.set_tensor(inp[0]["index"], img)
    interp.invoke()  # warmup
    times = []
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  threads=4, no explicit delegate: mean={sum(times)/len(times):.1f}ms")
except Exception as e:
    print(f"  Failed: {e}")

# Try 3: Check for XNNPackDelegate class
print("\n--- Check for XNNPack delegate class ---")
for attr in dir(ai_edge_litert):
    if 'xnn' in attr.lower() or 'delegate' in attr.lower():
        print(f"  ai_edge_litert.{attr}")

try:
    import ai_edge_litert.interpreter as interp_mod
    for attr in dir(interp_mod):
        if 'xnn' in attr.lower() or 'delegate' in attr.lower():
            print(f"  interpreter.{attr}")
except Exception:
    pass

# Try 4: Check if there's a way to use tf.lite's XNNPACK with ai_edge_litert ops
print("\n--- Try hybrid approach ---")
print("Import ai_edge_litert first (registers ops), then use tf.lite...")
try:
    # ai_edge_litert is already imported above (registered ops)
    from tensorflow.lite.python.interpreter import Interpreter as TfInterp
    interp = TfInterp(model_path=model_path, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    interp.set_tensor(inp[0]["index"], img)
    interp.invoke()  # warmup
    times = []
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  tf.lite after ai_edge_litert import: mean={sum(times)/len(times):.1f}ms")
except Exception as e:
    print(f"  Failed: {e}")
