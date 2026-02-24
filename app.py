import streamlit as st
import pandas as pd
import json, os
import tensorflow as tf
import numpy as np
from PIL import Image

st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff5f8, #eef5ff);
    color: #102a43;
    font-family: 'Poppins', sans-serif;
}

/* Title with black text and pink heart */
h1 {
    text-align: center;
    color: #000000;
    font-size: 2.4em;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-top: -10px;
    margin-bottom: 5px;
}

/* Info Box */
.info-box {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 14px 18px;
    margin: 12px 0;
    box-shadow: 0 4px 16px rgba(40, 40, 100, 0.1);
    backdrop-filter: blur(10px);
    text-align: center;
}
.info-title { font-size: 14px; color: #555; text-transform: uppercase; }
.info-value { font-size: 22px; color: #ff4b9a; font-weight: 700; }

/* Data Table */
.stDataFrame, table {
    border-radius: 10px !important;
    overflow: hidden !important;
}
thead tr th {
    background-color: #ff4b9a !important;
    color: white !important;
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 15px;
}
tbody tr:nth-child(odd) {
    background-color: #fff0f6 !important;
}
tbody tr:nth-child(even) {
    background-color: #ffffff !important;
}
tbody td {
    text-align: center !important;
    padding: 8px;
}
tbody tr:hover {
    background-color: #ffe3ef !important;
}

/* Upload Section */
.upload-box {
    border: 2px dashed #ff4b9a;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    text-align: center;
    transition: all 0.3s ease;
    font-weight: 500;
    font-size: 16px;
    color: #333;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(10, 100, 200, 0.1);
}
.upload-box:hover {
    border-color: #000000;
    background: rgba(255, 245, 250, 1);
}

/* Confidence Bar */
.conf-bar {
    width: 100%;
    background-color: #e9edff;
    border-radius: 8px;
    margin-top: 10px;
}
.conf-fill {
    height: 16px;
    border-radius: 8px;
    background: linear-gradient(90deg, #ff4b9a, #6a11cb);
}

/* Prediction Card */
.pred-card {
    text-align: center;
    padding: 20px;
    border-radius: 16px;
    font-weight: 600;
    font-size: 20px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 6px 18px rgba(12, 40, 80, 0.15);
}
.pred-benign { background: linear-gradient(120deg, #43cea2, #185a9d); }
.pred-malignant { background: linear-gradient(120deg, #ff416c, #ff4b2b); }

</style>
""", unsafe_allow_html=True)

#Title
st.markdown("<h1>🩷 Breast Cancer Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("Upload a histopathology image to classify it as **Benign** or **Malignant**, and view CNN & ML model performance metrics.")

#Dataset Info
dataset_dir = "dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
total_images = sum([len(files) for _, _, files in os.walk(dataset_dir)]) if os.path.exists(dataset_dir) else 0
st.markdown(f"""
<div class="info-box">
  <div class="info-title">Total Images in Dataset</div>
  <div class="info-value">{total_images:,} Images</div>
</div>
""", unsafe_allow_html=True)

#Model Comparison
if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    comparison_df = pd.DataFrame([{"Model": m, **vals} for m, vals in metrics.items()])
    st.subheader("📈 Model Performance Comparison")
    st.dataframe(comparison_df.round(2), use_container_width=True)
else:
    st.warning("⚠️ No metrics.json found. Please run training first.")

#Image Upload Section
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    model_path = "cnn_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        input_shape = model.input_shape[1:3]
        img_resized = img.resize(input_shape)
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        pred = model.predict(arr)[0][0]
        result = "Malignant" if pred > 0.5 else "Benign"
        confidence = float(pred if pred > 0.5 else 1 - pred) * 100

        #Prediction Card
        color_class = "pred-malignant" if result == "Malignant" else "pred-benign"
        st.markdown(f"""
        <div class="pred-card {color_class}">
            Prediction: <b>{result}</b><br>
            Confidence: {confidence:.2f}%
        </div>
        <div class="conf-bar"><div class="conf-fill" style="width:{confidence}%;"></div></div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ cnn_model.h5 not found. Please train the model first.")
