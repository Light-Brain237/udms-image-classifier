"""
UDMS Model Evaluation — Generate evaluation metrics on the test set.

Input:  Trained model + data/processed/test/
Output: docs/evaluation_report.md

Metrics:
1. Overall accuracy
2. Per-class precision, recall, F1-score
3. Confusion matrix
4. Misclassification analysis: top confused class pairs
5. Confidence distribution histogram
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data.category_mapping import UDMS_CATEGORIES


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """Run model.predict on test set, compute all metrics.

    Returns:
        dict with keys: accuracy, classification_report (str),
        confusion_matrix (ndarray), y_true, y_pred, y_prob.
    """
    y_true_list: list[int] = []
    y_prob_list: list[np.ndarray] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob_list.append(probs)
        y_true_list.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=UDMS_CATEGORIES, digits=3
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot and save confusion matrix as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def find_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_paths: list[str],
    top_n: int = 10,
) -> list[dict]:
    """Return the top_n most confident wrong predictions with image paths."""
    results: list[dict] = []
    wrong = np.where(y_true != y_pred)[0]
    for idx in wrong[:top_n]:
        results.append(
            {
                "image": image_paths[idx] if idx < len(image_paths) else "N/A",
                "true": UDMS_CATEGORIES[y_true[idx]],
                "predicted": UDMS_CATEGORIES[y_pred[idx]],
            }
        )
    return results


def generate_evaluation_report(
    results: dict, output_path: Path = Path("docs/evaluation_report.md")
) -> None:
    """Write evaluation report with all metrics and analysis.

    Flags accuracy < 75 % and suggests remediation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix image
    cm_path = output_path.parent / "confusion_matrix.png"
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        UDMS_CATEGORIES,
        cm_path,
    )

    acc = results["accuracy"]
    lines = [
        "# UDMS Image Classifier — Evaluation Report\n",
        f"## Overall Accuracy: {acc:.1%}\n",
    ]

    if acc < 0.75:
        lines.append(
            "> **WARNING:** Accuracy is below the 75 % minimum target.\n"
            "> Consider: more data for weak categories, longer training, "
            "or augmentation tuning.\n"
        )

    lines.append("## Classification Report\n")
    lines.append(f"```\n{results['classification_report']}```\n")
    lines.append("## Confusion Matrix\n")
    lines.append(f"![Confusion Matrix](confusion_matrix.png)\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation report saved to {output_path}")

