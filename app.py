import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("best_model.h5")

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown(
"""
<style>
    .title {
        text-align: center;
        color: #6C3483;
        font-family: 'Arial', sans-serif;
    }
    .prediction {
        text-align: center;
        font-size: 2em;
        margin-top: 20px;
    }
    .confidence {
        text-align: center;
        font-size: 1.5em;
        margin-top: 10px;
        color: #555;
    }
    .upload-btn {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        max-width: 400px;
        margin-left:auto;
        margin-right:auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🐱 Cat vs 🐶 Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

with st.container():
    uploaded_file = st.file_uploader("Upload an image of a cat or a dog", type=["jpg", "jpeg", "png"], label_visibility='hidden')
    upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
    with upload_col2:
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", width=300)

            with st.spinner('Classifying...'):
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]
                label = "Dog 🐶" if prediction >= 0.5 else "Cat 🐱"
                confidence = prediction if prediction >= 0.5 else 1 - prediction

            result_color = "#E74C3C" if label.startswith("Dog") else "#27AE60"
            st.markdown(
                f"""
                <div class='result-box' style='background-color: {result_color}4D;'>
                    <h2 class='prediction'>{label}</h2>
                    <p class='confidence'>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True
            )