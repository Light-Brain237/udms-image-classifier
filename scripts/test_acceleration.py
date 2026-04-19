"""Test ai_edge_litert compiled_model API and other acceleration options."""
import time
import numpy as np

model_path = "models/classifier.tflite"
img = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Test 1: ai_edge_litert with experimental_default_delegate_latest_features
print("--- Test: experimental_default_delegate_latest_features=True ---")
try:
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(
        model_path=model_path,
        num_threads=4,
        experimental_default_delegate_latest_features=True,
    )
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
    print(f"  mean={sum(times)/len(times):.1f}ms  {[f'{t:.0f}' for t in times]}")
except Exception as e:
    print(f"  Failed: {e}")

# Test 2: ai_edge_litert.compiled_model (new LiteRT API)
print("\n--- Test: compiled_model API ---")
try:
    from ai_edge_litert.compiled_model import CompiledModel
    cm = CompiledModel(model_path)
    print(f"  CompiledModel created successfully")
    print(f"  Methods: {[m for m in dir(cm) if not m.startswith('_')]}")
    # Try to run inference
    result = cm(img)
    # warmup done, benchmark
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = cm(img)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  mean={sum(times)/len(times):.1f}ms  {[f'{t:.0f}' for t in times]}")
except Exception as e:
    print(f"  Failed: {e}")

# Test 3: Check OpResolverType enum values
print("\n--- OpResolverType values ---")
try:
    from ai_edge_litert.interpreter import Interpreter
    import ai_edge_litert._pywrap_litert_interpreter_wrapper as wrapper
    for attr in dir(wrapper):
        if 'resolver' in attr.lower() or 'xnn' in attr.lower() or 'delegate' in attr.lower():
            print(f"  {attr}")
except Exception as e:
    print(f"  {e}")

# Test 4: Check the OpResolverType enum
print("\n--- Check OpResolverType enum ---")
try:
    from ai_edge_litert.interpreter import OpResolverType
    for item in OpResolverType:
        print(f"  {item.name} = {item.value}")
except Exception as e:
    print(f"  {e}")

# Test 5: Try each OpResolverType
print("\n--- Try different OpResolverTypes ---")
try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    for ort in OpResolverType:
        try:
            interp = Interpreter(
                model_path=model_path,
                num_threads=4,
                experimental_op_resolver_type=ort,
            )
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
            print(f"  {ort.name}: mean={sum(times)/len(times):.1f}ms")
        except Exception as e:
            print(f"  {ort.name}: Failed - {e}")
except Exception as e:
    print(f"  {e}")
