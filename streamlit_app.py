"""
Handwritten Alphabet Recognition - Streamlit App
Innomatics Research Labs - Image Classification Capstone Project

Run this app using:
    streamlit run streamlit_app.py
"""

import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

TRAINING_IMAGE_SIZE = 34


def extract_features(image_34x34):
    img_array = np.array(image_34x34) / 255.0
    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def predict_letter(image):
    image = image.convert("L").resize((TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE))
    features = extract_features(image).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_letter = label_encoder.inverse_transform([prediction])[0]
    confidence = round(max(probabilities) * 100, 2)

    prob_dict = {
        label_encoder.classes_[i]: round(probabilities[i] * 100, 2)
        for i in range(len(label_encoder.classes_))
    }

    return predicted_letter, confidence, prob_dict


st.title("Handwritten Alphabet Recognition")
st.write(
    "Upload an image of a single handwritten letter (a white stroke on a "
    "black background works best, matching the training data style)."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Predict"):
        letter, confidence, prob_dict = predict_letter(image)

        st.subheader("Prediction Result")
        st.success(f"Predicted Letter: **{letter}**")
        st.write(f"**Confidence Score:** {confidence}%")

        st.subheader("Top 5 Predictions")
        top_5 = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        st.bar_chart(top_5)

st.markdown("---")
st.caption("Image Classification Capstone Project - Innomatics Research Labs")
