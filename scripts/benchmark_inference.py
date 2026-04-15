"""
UDMS Inference Benchmark — Measure TFLite inference performance.

Metrics:
- Average inference time (ms)
- P50, P95, P99 latency
- Throughput (images/sec)
"""

import time
import statistics
from pathlib import Path

import numpy as np


def benchmark_tflite(
    model_path: str = "models/classifier.tflite",
    num_runs: int = 100,
) -> dict:
    """Benchmark TFLite inference speed with a dummy image."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create a dummy input matching model's expected shape
    input_shape = input_details[0]["shape"]
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    times.sort()
    results = {
        "num_runs": num_runs,
        "avg_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * len(times))],
        "p99_ms": times[int(0.99 * len(times))],
        "min_ms": min(times),
        "max_ms": max(times),
        "throughput_ips": 1000.0 / statistics.mean(times),
    }

    return results


def main():
    model_path = "models/classifier.tflite"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Export the model first: python -m src.training.export")
        return

    print("=" * 60)
    print("UDMS TFLite Inference Benchmark")
    print("=" * 60)

    results = benchmark_tflite(model_path, num_runs=100)

    print(f"\nRuns:       {results['num_runs']}")
    print(f"Average:    {results['avg_ms']:.1f} ms")
    print(f"Median:     {results['median_ms']:.1f} ms")
    print(f"P95:        {results['p95_ms']:.1f} ms")
    print(f"P99:        {results['p99_ms']:.1f} ms")
    print(f"Min:        {results['min_ms']:.1f} ms")
    print(f"Max:        {results['max_ms']:.1f} ms")
    print(f"Throughput: {results['throughput_ips']:.1f} images/sec")

    if results["avg_ms"] < 500:
        print("\n✅ PASS: Average inference < 500ms")
    else:
        print("\n❌ FAIL: Average inference >= 500ms — optimization needed")

    if results["avg_ms"] < 200:
        print("✅ STRETCH: Average inference < 200ms")


if __name__ == "__main__":
    main()
