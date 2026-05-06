"""
UDMS Image Classifier — Streamlit Demo App

Allows anyone to upload a photo and see the AI classification result
visually — no technical knowledge needed.

Run:
    streamlit run demo/streamlit_app.py
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Paths are resolved relative to the project root (one level up from demo/)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "classifier.tflite"
LABEL_MAP_PATH = PROJECT_ROOT / "models" / "label_map.json"

IMAGE_SIZE = 224          # Model input: 224×224 pixels
MAX_UPLOAD_MB = 10        # Reject files larger than this
REVIEW_THRESHOLD = 0.6    # Confidence below this triggers a human-review warning

# Human-readable descriptions shown in the header
CATEGORY_DESCRIPTIONS = {
    "bad_drainage":           "Bad Drainage / Water Sewage Issues",
    "damaged_signage":        "Damaged Signage / Infrastructure",
    "illegal_dumping":        "Illegal Dumping / Garbage",
    "potholes":               "Pothole / Road Damage",
    "vegetation_overgrowth":  "Vegetation Overgrowth",
}


# ---------------------------------------------------------------------------
# Model loading — cached so the interpreter is created only once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the TFLite model using ai_edge_litert and return the interpreter.

    @st.cache_resource ensures this runs only once even across re-runs.
    """
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        experimental_default_delegate_latest_features=True,
    )
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_label_map() -> dict:
    """Load the index-to-category mapping from label_map.json."""
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert string keys to int: {0: {"category": ..., "label": ...}, ...}
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize a PIL image and return raw float32 pixels in [0, 255].

    The MobileNetV2 Lambda layer baked into the TFLite model handles the
    [-1, 1] scaling internally — do NOT normalise here or predictions will
    be wrong.

    Steps:
      1. Convert to RGB (handles grayscale and RGBA uploads).
      2. Resize to IMAGE_SIZE × IMAGE_SIZE using LANCZOS resampling.
      3. Cast to float32 — values stay in [0, 255].
      4. Add batch dimension → shape (1, 224, 224, 3).
    """
    img = pil_image.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # raw [0, 255] — model scales internally
    return np.expand_dims(arr, axis=0)      # add batch dim


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(interpreter, image_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Feed one image through the TFLite model and return (scores, elapsed_ms).

    Returns:
        scores      — float32 array of shape (5,), one probability per class.
        elapsed_ms  — wall-clock inference time in milliseconds.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image_array)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    scores = interpreter.get_tensor(output_details[0]["index"])[0]  # shape (5,)
    return scores, elapsed_ms


# ---------------------------------------------------------------------------
# Results chart
# ---------------------------------------------------------------------------

def make_bar_chart(scores: np.ndarray, label_map: dict) -> plt.Figure:
    """Return a horizontal bar chart showing all 5 category probabilities."""
    labels = [label_map[i]["label"] for i in range(len(scores))]
    values = scores.tolist()

    # Colour each bar by confidence level
    colours = []
    for v in values:
        if v >= 0.8:
            colours.append("#2ecc71")   # green  — high confidence
        elif v >= 0.6:
            colours.append("#f39c12")   # orange — medium confidence
        else:
            colours.append("#e74c3c")   # red    — low confidence

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels, values, color=colours)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("All Category Scores")

    # Annotate each bar with its percentage
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Populate the sidebar with model info and an About section."""
    with st.sidebar:
        st.header("Model Information")
        st.markdown(
            """
| Field | Value |
|---|---|
| **Architecture** | MobileNetV2 |
| **Categories** | 5 |
| **Test Accuracy** | 84.78 % |
| **Model size** | 2.55 MB |
| **Inference speed** | ~141 ms |
"""
        )

        st.divider()

        st.header("About UDMS")
        st.markdown(
            """
The **Urban Disorder Monitoring System (UDMS)** uses AI to automatically
classify photos of urban issues reported by the public.

This helps municipalities **prioritise repairs** and **allocate resources**
faster — without manual triage of every photo.

**Supported issue types:**
- Bad drainage & flooding
- Damaged signs & infrastructure
- Illegal dumping & litter
- Potholes & road damage
- Vegetation overgrowth
"""
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="UDMS Image Classifier Demo",
        page_icon="🏙️",
        layout="wide",
    )

    render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🏙️ UDMS Image Classifier Demo")
    st.subheader("Upload a photo of urban disorder and see the AI classify it instantly")

    # Show the 5 supported categories as info chips
    st.markdown("**Supported categories:**")
    cols = st.columns(5)
    for col, (key, desc) in zip(cols, CATEGORY_DESCRIPTIONS.items()):
        col.info(f"**{key}**\n\n{desc}")

    st.divider()

    # ── Upload section ───────────────────────────────────────────────────────
    st.markdown("### 📤 Upload an Image")
    uploaded = st.file_uploader(
        "Choose a photo (JPG, JPEG, PNG, or WEBP — max 10 MB)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("Waiting for an image upload…")
        return

    # Enforce file size limit
    file_bytes = uploaded.getvalue()
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")
        return

    # Display the uploaded image
    pil_image = Image.open(uploaded)
    left_col, _ = st.columns([1, 1])
    with left_col:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    # ── Classification ────────────────────────────────────────────────────────
    with st.spinner("Classifying image…"):
        interpreter = load_model()
        label_map   = load_label_map()
        image_array = preprocess_image(pil_image)
        scores, elapsed_ms = run_inference(interpreter, image_array)

    top_idx        = int(np.argmax(scores))
    confidence     = float(scores[top_idx])
    category       = label_map[top_idx]["category"]
    category_label = label_map[top_idx]["label"]
    requires_review = confidence < REVIEW_THRESHOLD

    # ── Results display ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Classification Result")

    # Colour-coded confidence badge
    if confidence >= 0.8:
        badge_colour = "green"
        confidence_label = "High Confidence"
    elif confidence >= 0.6:
        badge_colour = "orange"
        confidence_label = "Medium Confidence"
    else:
        badge_colour = "red"
        confidence_label = "Low Confidence"

    result_col, chart_col = st.columns([1, 1])

    with result_col:
        st.markdown(f"## **{category_label}**")
        st.markdown(f"*Category key: `{category}`*")

        st.markdown(f"**:{badge_colour}[{confidence_label}]**")
        st.progress(confidence, text=f"{confidence:.1%}")

        st.metric(label="Confidence Score", value=f"{confidence:.1%}")
        st.caption(f"⏱ Inference time: **{elapsed_ms:.0f} ms**")

        if requires_review:
            st.warning(
                "⚠️ **Requires Human Review** — confidence is below 60 %. "
                "Please verify this classification manually."
            )

    with chart_col:
        st.markdown("**All category scores**")
        fig = make_bar_chart(scores, label_map)
        st.pyplot(fig)
        plt.close(fig)   # free memory after rendering


if __name__ == "__main__":
    main()

