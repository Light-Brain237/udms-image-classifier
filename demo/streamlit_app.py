"""
UDMS Image Classifier — Streamlit Demo App

Upload an image → see prediction with confidence bar + category label.

Run:
    streamlit run demo/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.classifier import UDMSClassifier

MODEL_PATH = "models/classifier.tflite"
LABEL_MAP_PATH = "models/label_map.json"


@st.cache_resource
def load_classifier():
    return UDMSClassifier(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        confidence_threshold=0.6,
    )


def main():
    st.set_page_config(page_title="UDMS Image Classifier Demo", layout="centered")
    st.title("UDMS Image Classifier Demo")
    st.markdown("Upload a photo of an urban disorder issue to classify it.")

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

        image_bytes = uploaded.getvalue()

        with st.spinner("Classifying..."):
            classifier = load_classifier()
            result = classifier.predict(image_bytes)

        pred = result["prediction"]

        st.subheader(f"Prediction: {pred['category_label']}")
        st.progress(pred["confidence"])
        st.metric("Confidence", f"{pred['confidence']:.1%}")

        if pred["requires_review"]:
            st.warning("Low confidence — manual review recommended.")

        st.markdown("### Alternatives")
        for alt in result.get("alternatives", []):
            st.write(f"- {alt['category']}: {alt['confidence']:.1%}")

        st.caption(f"Inference time: {result['inference_time_ms']:.0f} ms")


if __name__ == "__main__":
    main()
