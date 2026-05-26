import streamlit as st
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
BACKGROUND
========================================================= */

.stApp{

    background:
    linear-gradient(
        135deg,
        #8de08f 0%,
        #77d5e7 45%,
        #a46bd6 100%
    );

    background-attachment: fixed;

    color:white;
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.block-container{

    max-width:1350px;

    padding-top:1.5rem;

    padding-bottom:2rem;
}

/* =========================================================
HEADER
========================================================= */

.main-title{

    text-align:center;

    font-size:64px;

    font-weight:900;

    color:white;

    margin-bottom:10px;

    letter-spacing:-1px;

    text-shadow:
    0px 5px 20px rgba(0,0,0,0.35);
}

.sub-title{

    text-align:center;

    font-size:22px;

    color:#f1f5f9;

    margin-bottom:45px;

    font-weight:500;
}

/* =========================================================
GLASS CARDS
========================================================= */

.card{

    background:rgba(15,23,42,0.72);

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    -webkit-backdrop-filter: blur(18px);

    border-radius:28px;

    padding:30px;

    margin-bottom:25px;

    box-shadow:
    0px 10px 35px rgba(0,0,0,0.35);

    transition:0.35s ease;
}

/* Hover effect */

.card:hover{

    transform:translateY(-6px);

    box-shadow:
    0px 18px 45px rgba(0,0,0,0.45);
}

/* =========================================================
HEADINGS
========================================================= */

.card h1,
.card h2,
.card h3{

    color:#7dd3fc;

    margin-bottom:18px;

    font-weight:800;
}

/* =========================================================
TEXT
========================================================= */

.card p,
.card li{

    color:#f8fafc;

    font-size:16px;

    line-height:1.9;
}

/* =========================================================
UPLOAD BOX
========================================================= */

[data-testid="stFileUploader"]{

    background:rgba(15,23,42,0.72);

    border:2px dashed rgba(255,255,255,0.2);

    border-radius:22px;

    padding:22px;

    transition:0.3s;
}

[data-testid="stFileUploader"]:hover{

    border:2px dashed #38bdf8;

    box-shadow:
    0px 0px 20px rgba(56,189,248,0.4);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button{

    width:100%;

    height:58px;

    border:none;

    border-radius:18px;

    background:
    linear-gradient(
        135deg,
        #0f172a,
        #1d4ed8
    );

    color:white;

    font-size:18px;

    font-weight:700;

    transition:0.35s ease;

    box-shadow:
    0px 6px 20px rgba(30,64,175,0.5);
}

/* Hover */

.stButton > button:hover{

    transform:translateY(-3px);

    background:
    linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb
    );

    box-shadow:
    0px 10px 30px rgba(37,99,235,0.7);
}

/* =========================================================
METRICS
========================================================= */

[data-testid="metric-container"]{

    background:rgba(15,23,42,0.82);

    border:1px solid rgba(255,255,255,0.08);

    padding:20px;

    border-radius:22px;

    box-shadow:
    0px 6px 20px rgba(0,0,0,0.35);

    transition:0.3s;
}

/* Hover */

[data-testid="metric-container"]:hover{

    transform:translateY(-4px);
}

/* Metric text */

[data-testid="metric-container"] label{

    color:#cbd5e1 !important;

    font-size:16px;
}

[data-testid="metric-container"] div{

    color:white !important;

    font-size:28px;
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        rgba(15,23,42,0.96),
        rgba(30,41,59,0.96)
    );
}

/* =========================================================
IMAGE
========================================================= */

img{

    border-radius:20px;
}

/* =========================================================
ALERTS
========================================================= */

.stSuccess,
.stWarning,
.stError,
.stInfo{

    border-radius:18px;
}

/* =========================================================
FOOTER
========================================================= */

.footer{

    text-align:center;

    margin-top:50px;

    color:#e2e8f0;

    font-size:15px;
}

/* =========================================================
HIDE STREAMLIT
========================================================= */

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

header{
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
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AI-Based Road Damage Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Smart City Infrastructure Monitoring using CNN</div>',
    unsafe_allow_html=True
)

# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

left_col, right_col = st.columns([1.2,1])

# ============================================================
# LEFT SIDE
# ============================================================

with left_col:

    st.markdown("""
    <div class="card">

    <h2>About the Project</h2>

    <p>
    Road monitoring is extremely important for public safety,
    smart transportation systems, and infrastructure management.
    Delayed detection of road damage can increase:
    </p>

    <ul>
        <li>Road accidents</li>
        <li>Vehicle damage</li>
        <li>Traffic congestion</li>
        <li>Maintenance costs</li>
    </ul>

    <h3>Role of CNN in Computer Vision</h3>

    <p>
    Convolutional Neural Networks (CNNs) automatically learn
    road surface features such as cracks, potholes,
    and texture patterns for intelligent classification.
    </p>

    <h3>Industry Applications</h3>

    <ul>
        <li>Smart City Monitoring</li>
        <li>Road Safety Systems</li>
        <li>Municipal Infrastructure Analysis</li>
        <li>Autonomous Vehicle Navigation</li>
        <li>Highway Inspection Automation</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RIGHT SIDE
# ============================================================

with right_col:

    st.markdown("""
    <div class="card">

    <h2>Upload Road Image</h2>

    <p>
    Upload a road surface image for AI-powered
    road damage analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # ========================================================
    # PREPROCESS
    # ========================================================

    IMG_SIZE = 128

    img = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    label = index_to_label[predicted_class]

    # ========================================================
    # SEVERITY
    # ========================================================

    label_lower = label.lower()

    if "pothole" in label_lower:

        severity = "High"

        recommendation = """
        Immediate maintenance recommended.
        High-risk road condition detected.
        """

    elif "crack" in label_lower:

        severity = "Medium"

        recommendation = """
        Scheduled maintenance recommended.
        Surface deterioration detected.
        """

    else:

        severity = "Low"

        recommendation = """
        Routine inspection recommended.
        Moderate infrastructure issue detected.
        """

    # ========================================================
    # IMAGE + RESULTS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    img_col, result_col = st.columns([1,1])

    # ========================================================
    # IMAGE COLUMN
    # ========================================================

    with img_col:

        st.markdown("""
        <div class="card">
        <h2>Uploaded Image Preview</h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # ========================================================
    # RESULTS COLUMN
    # ========================================================

    with result_col:

        st.markdown("""
        <div class="card">
        <h2>Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Prediction",
            label
        )

        metric2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        metric3.metric(
            "Severity",
            severity
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ====================================================
        # VISUALIZATION
        # ====================================================

        st.markdown("""
        <div class="card">
        <h2>Class Confidence Graph</h2>
        </div>
        """, unsafe_allow_html=True)

        class_names = list(index_to_label.values())

        probabilities = prediction[0] * 100

        fig, ax = plt.subplots(figsize=(8,4))

        bars = ax.bar(
            class_names,
            probabilities
        )

        ax.set_ylabel("Confidence (%)")

        ax.set_xlabel("Damage Classes")

        ax.set_title("Prediction Probability Chart")

        ax.set_facecolor("#0f172a")

        fig.patch.set_facecolor("#0f172a")

        ax.tick_params(colors='white')

        ax.yaxis.label.set_color('white')

        ax.xaxis.label.set_color('white')

        ax.title.set_color('white')

        st.pyplot(fig)

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(f"""
    <div class="card">

    <h2>Maintenance Recommendations</h2>

    <p>
    <b>Repair Priority:</b> {severity}
    </p>

    <p>
    {recommendation}
    </p>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Developed using CNN • Keras • Streamlit</div>',
    unsafe_allow_html=True
)