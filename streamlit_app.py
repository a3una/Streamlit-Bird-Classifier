"""Free-deployable web app for the COS30082 CUB-200 classifier.

Run locally:
    streamlit run streamlit_app.py
Deploy:
    Streamlit Community Cloud, with this file as the entry point.
"""
from pathlib import Path
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.inference import load_classifier, predict, species_names

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "best_model.pth"
LABEL_MAP = ROOT / "class_names.csv"

st.set_page_config(
    page_title="Bird Species Classifier · COS30082",
    page_icon="🐦",
    layout="centered",
)

st.title("🐦 Bird Species Classifier")
st.caption("COS30082 • 200-class fine-tuned PyTorch image classifier")
st.write("Upload a photo to view the five most likely predicted classes.")


@st.cache_resource(show_spinner="Loading the trained EfficientNet-B0 model…")
def get_model():
    """Cache model weights across visits to avoid reloading every time."""
    return load_classifier(WEIGHTS)


if not WEIGHTS.is_file():
    st.error("Missing models/best_model.pth")
    st.info(
        "Copy your saved models/best_model.pth into this project's models/ folder, "
        "commit it to your GitHub repository, then redeploy."
    )
    st.stop()

try:
    model, preprocess, checkpoint = get_model()
except Exception as exc:
    st.error("Could not load the model checkpoint.")
    st.exception(exc)
    st.stop()

labels = species_names(checkpoint, LABEL_MAP)
upload = st.file_uploader("Choose a JPG or PNG bird photo", type=["jpg", "jpeg", "png"])

if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()
    st.image(image, caption=upload.name, use_container_width=True)
    with st.spinner("Identifying the bird…"):
        results = predict(image, model, preprocess, labels, top_k=5)

    st.subheader("Predictions")
    for rank, (class_id, display_name, probability) in enumerate(results, 1):
        st.markdown(f"**{rank}. {display_name}** — Class ID `{class_id}` · {probability:.1%}")
        st.progress(float(probability))

    if results and any(str(label).startswith("Class ") for _, label, _ in results):
        st.info(
            "Your checkpoint uses numeric/image filenames instead of real species names. "
            "Class IDs are accurate; to show names, add class_names.csv with "
            "columns class_id,species_name in the correct training-label order."
        )
    st.caption("Scores are model probabilities, not a guarantee that a prediction is correct.")

with st.expander("Model information"):
    st.write(f"**Architecture:** {checkpoint['model_name']}")
    st.write(f"**Classes:** {checkpoint['num_classes']}")
    st.write(f"**Input size:** {checkpoint['image_size']} × {checkpoint['image_size']}")
    st.write("**Inference hardware:** CPU (no paid GPU required)")

st.divider()
st.caption("Course assignment demo. The provided test data was used only for final evaluation.")
