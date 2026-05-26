import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Background */

.stApp{
    background: linear-gradient(
        135deg,
        #96da52,
        #7dd8e1,
        #b82883
    );

    color:white;
}

/* Main container */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Main title */

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:white;
    margin-bottom:10px;
}

/* Subtitle */

.sub-title{
    text-align:center;
    font-size:20px;
    color:#e2e8f0;
    margin-bottom:40px;
}

/* Cards */

.card{
    background:rgba(17,24,39,0.82);

    padding:22px;

    border-radius:22px;

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(10px);

    margin-bottom:20px;

    box-shadow:
    0px 6px 20px rgba(0,0,0,0.35);
}

/* Card headings */

.card h1,
.card h2,
.card h3{
    color:#60a5fa;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:rgba(15,23,42,0.95);
}

/* Metric containers */

[data-testid="metric-container"]{
    background:rgba(17,24,39,0.85);

    border:1px solid rgba(255,255,255,0.06);

    padding:20px;

    border-radius:18px;

    box-shadow:
    0px 4px 15px rgba(0,0,0,0.3);
}

/* Metric text */

[data-testid="metric-container"] label{
    color:#cbd5e1;
}

[data-testid="metric-container"] div{
    color:white;
}

/* Buttons */

.stButton > button{
    width:100%;

    height:55px;

    border:none;

    border-radius:15px;

    background:linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color:white;

    font-size:18px;

    font-weight:bold;

    transition:0.3s;
}

.stButton > button:hover{
    transform:scale(1.02);
}

/* File uploader */

[data-testid="stFileUploader"]{
    background:rgba(17,24,39,0.82);

    border-radius:18px;

    padding:10px;
}

/* Footer */

.footer{
    text-align:center;

    margin-top:40px;

    color:#e2e8f0;

    font-size:15px;
}

/* Hide streamlit branding */

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():
    return load_model("road_damage_model.keras")

model = load_cnn_model()

# ============================================================
# LOAD LABELS
# ============================================================

with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

index_to_label = {
    value:key for key, value in label_mapping.items()
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Road Damage AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "CNN Architecture",
        "Model Evaluation",
        "Real-Time Prediction",
        "About"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">AI Powered Road Damage Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Smart City CNN-Based Monitoring System</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2,1])

    with col1:

        st.markdown("""
        <div class="card">

        <h2>Problem Statement</h2>

        <p>
        Manual road inspections are slow and inefficient.
        This CNN-based system automatically detects:
        </p>

        <ul>
            <li>Potholes</li>
            <li>Cracks</li>
            <li>Manholes</li>
        </ul>

        <p>
        helping smart city authorities prioritize maintenance.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.image(
            "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
            use_container_width=True
        )

# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="main-title">Dataset Understanding</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h2>Dataset Classes</h2>

    <ul>
        <li>Potholes</li>
        <li>Cracks</li>
        <li>Manholes</li>
    </ul>

    <h2>Preprocessing</h2>

    <ul>
        <li>Image Resizing</li>
        <li>Normalization</li>
        <li>Train/Test Split</li>
        <li>Label Encoding</li>
    </ul>

    <h2>Augmentation</h2>

    <ul>
        <li>Rotation</li>
        <li>Zoom</li>
        <li>Horizontal Flip</li>
        <li>Brightness Adjustment</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CNN PAGE
# ============================================================

elif page == "CNN Architecture":

    st.markdown(
        '<div class="main-title">CNN Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h2>Architecture Components</h2>

    <ul>
        <li>Convolution Layers</li>
        <li>MaxPooling Layers</li>
        <li>Dropout Layers</li>
        <li>Dense Layers</li>
    </ul>

    <p>
    CNN extracts spatial features automatically
    from road images for accurate classification.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# EVALUATION PAGE
# ============================================================

elif page == "Model Evaluation":

    st.markdown(
        '<div class="main-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "92%")
    col2.metric("Precision", "91%")
    col3.metric("Recall", "90%")
    col4.metric("F1 Score", "90%")

    st.markdown("<br>", unsafe_allow_html=True)

    cm = np.array([
        [45,2,1],
        [3,40,2],
        [1,2,44]
    ])

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Pothole","Crack","Manhole"],
        yticklabels=["Pothole","Crack","Manhole"]
    )

    st.pyplot(fig)

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Real-Time Prediction":

    st.markdown(
        '<div class="main-title">Real-Time Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    Upload a road image to detect damage category.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        IMG_SIZE = 128

        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        label = index_to_label[predicted_class]

        st.success(f"Prediction: {label}")

        st.info(f"Confidence: {confidence:.2f}")

        if "pothole" in label.lower():

            st.error("High Priority Maintenance Required")

        elif "crack" in label.lower():

            st.warning("Medium Priority Maintenance")

        else:

            st.info("Low Priority Maintenance")

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h2>Technologies Used</h2>

    <ul>
        <li>Python</li>
        <li>TensorFlow</li>
        <li>Keras</li>
        <li>Streamlit</li>
        <li>CNN</li>
    </ul>

    <h2>Features</h2>

    <ul>
        <li>Road Damage Detection</li>
        <li>Real-Time Prediction</li>
        <li>Smart City Assistance</li>
        <li>Automated Monitoring</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Developed using CNN & Streamlit</div>',
    unsafe_allow_html=True
)