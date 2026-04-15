"""
UDMS Classifier — Production inference wrapper around TFLite model.

Loaded ONCE at startup, reused for every request.
Uses src.data.preprocessing (no duplicate preprocessing module).
"""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_bytes, add_batch_dimension
from src.data.category_mapping import CATEGORY_LABELS


class UDMSClassifier:
    """Wraps a TFLite model for single-image classification."""

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Load TFLite model and label map once.

        Args:
            model_path: Path to ``classifier.tflite``.
            label_map_path: Path to ``label_map.json``.
            confidence_threshold: Below this confidence → ``requires_review``.
        """
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        with open(label_map_path, encoding="utf-8") as f:
            self.label_map: dict = json.load(f)

        self.confidence_threshold = confidence_threshold
        self._model_path = model_path

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes.

        Returns:
            {
                "prediction": {
                    "category": str,
                    "category_label": str,
                    "confidence": float,
                    "requires_review": bool,
                },
                "alternatives": [{"category": str, "confidence": float}, ...],
                "model_version": "1.0.0",
                "inference_time_ms": float,
            }
        """
        start = time.perf_counter()

        processed = preprocess_bytes(image_bytes)
        input_data = add_batch_dimension(processed)

        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_data
        )
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        entry = self.label_map[str(top_idx)]

        return {
            "prediction": {
                "category": entry["category"],
                "category_label": entry["label"],
                "confidence": confidence,
                "requires_review": confidence < self.confidence_threshold,
            },
            "alternatives": self._get_alternatives(probs),
            "model_version": "1.0.0",
            "inference_time_ms": round(elapsed_ms, 1),
        }

    def _get_alternatives(
        self, probabilities: np.ndarray, top_n: int = 3
    ) -> list[dict]:
        """Return top_n predictions excluding the primary, sorted desc."""
        indices = np.argsort(probabilities)[::-1]
        results: list[dict] = []
        for idx in indices[1 : top_n + 1]:
            entry = self.label_map[str(int(idx))]
            results.append(
                {
                    "category": entry["category"],
                    "confidence": round(float(probabilities[idx]), 4),
                }
            )
        return results

    @property
    def model_info(self) -> dict:
        """Return model metadata."""
        input_shape = list(self._input_details[0]["shape"])
        model_size = Path(self._model_path).stat().st_size
        return {
            "version": "1.0.0",
            "categories": [
                self.label_map[str(i)]["category"]
                for i in range(len(self.label_map))
            ],
            "input_shape": input_shape[1:],  # drop batch dim
            "model_size_mb": round(model_size / 1024 / 1024, 2),
        }

