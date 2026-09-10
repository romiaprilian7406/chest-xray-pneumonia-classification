from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.5


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        safe_mode=False,
        custom_objects={"rgb_to_grayscale": tf.image.rgb_to_grayscale},
    )


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


st.set_page_config(page_title="Deteksi Pneumonia X-Ray", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, [data-testid="stToolbar"], footer, [data-testid="stDecoration"] {
        visibility: hidden;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #22c55e !important;
        color: white !important;
        border: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #16a34a !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        margin-top: 0.5rem;
    }
    .result-card.pneumonia {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .result-card.normal {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .result-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-label.pneumonia { color: #b91c1c; }
    .result-label.normal { color: #15803d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Deteksi Pneumonia dari X-Ray Dada")
st.write("Upload citra X-ray dada untuk mendeteksi indikasi Pneumonia.")

uploaded_file = st.file_uploader("Upload citra X-ray dada (JPG/JPEG/PNG) - 200MB per file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Menganalisis..."):
        model = load_model()
        prob = float(model.predict(preprocess(image), verbose=0)[0][0])

    label = CLASSES[1] if prob > THRESHOLD else CLASSES[0]
    confidence = prob if label == "PNEUMONIA" else 1 - prob
    css_class = "pneumonia" if label == "PNEUMONIA" else "normal"

    col_image, col_result = st.columns(2)

    with col_image:
        st.image(image, caption="Citra yang diupload", use_container_width=True)

    with col_result:
        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="result-label {css_class}">{label}</div>
                <div>Confidence: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence)
