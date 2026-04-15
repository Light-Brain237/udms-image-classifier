"""
UDMS Model Export — Convert trained Keras model to TFLite and ONNX.

Input:  models/phase2_best.h5
Output: models/classifier.tflite, models/classifier.onnx, models/label_map.json

TFLite: dynamic-range quantisation (INT8 weights, float32 I/O).
ONNX:   via tf2onnx CLI.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.category_mapping import UDMS_CATEGORIES, CATEGORY_LABELS


def export_tflite(
    model_path: str,
    output_path: str = "models/classifier.tflite",
    quantize: bool = True,
) -> int:
    """Convert Keras .h5 to TFLite. Return file size in bytes.

    If ``quantize=True``, applies dynamic-range quantisation (INT8 weights).
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)

    size = out.stat().st_size
    print(f"TFLite saved to {out}  ({size / 1024 / 1024:.1f} MB)")
    return size


def export_onnx(
    model_path: str, output_path: str = "models/classifier.onnx"
) -> int:
    """Convert Keras .h5 to ONNX via tf2onnx. Return file size in bytes."""
    # tf2onnx needs a SavedModel directory
    tmp_saved = Path("models/_tmp_saved_model")
    model = tf.keras.models.load_model(model_path)
    model.save(str(tmp_saved))

    subprocess.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(tmp_saved),
            "--output",
            output_path,
        ],
        check=True,
    )

    # Clean up temp SavedModel
    import shutil

    shutil.rmtree(tmp_saved, ignore_errors=True)

    size = Path(output_path).stat().st_size
    print(f"ONNX saved to {output_path}  ({size / 1024 / 1024:.1f} MB)")
    return size


def create_label_map(
    output_path: str = "models/label_map.json",
) -> None:
    """Write label_map.json with category and human-readable label per index."""
    label_map = {
        str(i): {"category": cat, "label": CATEGORY_LABELS[cat]}
        for i, cat in enumerate(UDMS_CATEGORIES)
    }
    Path(output_path).write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    print(f"Label map saved to {output_path}")


def benchmark_tflite(
    tflite_path: str,
    sample_image: np.ndarray,
    num_runs: int = 100,
) -> float:
    """Run inference ``num_runs`` times; return average time in ms.

    Must be < 500 ms on CPU.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(sample_image, axis=0).astype(np.float32)

    # Warm-up
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"TFLite benchmark: {avg:.1f} ms avg over {num_runs} runs")
    return avg


def export_all(model_path: str = "models/phase2_best.h5") -> None:
    """Orchestrate: TFLite export, ONNX export, label map, benchmark."""
    tflite_size = export_tflite(model_path)
    assert tflite_size < 50 * 1024 * 1024, "TFLite model exceeds 50 MB!"

    try:
        export_onnx(model_path)
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")

    create_label_map()

    # Benchmark with a dummy image
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    avg_ms = benchmark_tflite("models/classifier.tflite", dummy)
    assert avg_ms < 500, f"Inference too slow: {avg_ms:.0f} ms (limit 500 ms)"

    print("\nAll exports complete.")


if __name__ == "__main__":
    export_all()

