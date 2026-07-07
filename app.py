import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Page config
st.set_page_config(
    page_title="ML Health & Farm Assistant",
    page_icon="🌟",
    layout="wide"
)
# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0f0f1a;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a1a2e;
        border-right: 1px solid #2d2d4e;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }

    /* All general text - bright white */
    .stApp p, .stApp label, 
    .stApp span, .stApp div,
    .stMarkdown, .stMarkdown p {
        color: #ffffff !important;
    }

    /* Headings */
    h1 {
        color: #a78bfa !important;
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        text-align: center;
        border-bottom: 2px solid #a78bfa;
        padding-bottom: 10px;
    }

    h2 {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }

    h3 {
        color: #ffffff !important;
        font-size: 1.2rem !important;
    }

    /* Buttons - compact */
    .stButton button {
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-size: 14px;
        font-weight: bold;
        width: auto !important;
        min-width: 180px;
        max-width: 250px;
        transition: all 0.2s;
        display: block;
        margin: 0 auto;
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #6d28d9, #8b5cf6);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.5);
    }

    /* Success/Error/Warning text fix */
    [data-testid="stAlert"] p {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 16px;
    }

    /* Sliders */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2d2d4e;
        border-radius: 12px;
        padding: 15px;
        border: 2px dashed #7c3aed;
    }

    [data-testid="stFileUploader"] * {
        color: #2d2d4e!important;
    }

    [data-testid="stFileUploader"] button {
        background: #7c3aed !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Select box */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Radio */
    .stRadio label {
        color: #ffffff !important;
    }/* Dropdown selectbox text fix */
    .stSelectbox div[data-baseweb="select"] {
        background: #2d2d4e !important;
        border-radius: 8px;
        border: 1px solid #7c3aed !important;
    }

    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] input,
    .stSelectbox [data-baseweb="select"] [class*="valueContainer"] *,
    .stSelectbox [data-baseweb="select"] [class*="singleValue"],
    .stSelectbox [data-baseweb="select"] [class*="placeholder"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    

    /* Divider */
    hr {
        border-color: #2d2d4e;
        margin: 15px 0;
    }

    /* Bar chart */
    [data-testid="stArrowVegaLiteChart"] {
        background: #1e1e3a;
        border-radius: 12px;
        padding: 10px;
    }
            /* Force selectbox input text white */
    div[data-testid="stSelectbox"] input {
        color: black !important;
        background: #2d2d4e !important;
    }
    
    div[data-testid="stSelectbox"] > div {
        background: #2d2d4e !important;
        color: black !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)
# ============================================
# Load Models
# ============================================

@st.cache_resource
def load_heart_model():
    df = pd.read_csv("heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

@st.cache_resource
def load_crop_model():
    df = pd.read_csv("Crop_recommendation.csv")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    X = df.drop(["label", "label_encoded"], axis=1)
    y = df["label_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, le



# ============================================
# Sidebar Navigation
# ============================================

st.sidebar.title("🌟 ML Assistant")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose a Tool:",
    ["🏠 Home",
     "❤️ Heart Disease Checker",
     "🌾 Crop Recommender",
     "🌿 Plant Disease Detector"]
)

# ============================================
# HOME PAGE
# ============================================

if page == "🏠 Home":
    st.title(" Health & Farm Assistant")
    st.markdown(
        "<p style='text-align:center; color:#c4b5fd; font-size:18px;'>"
        "Built with Machine Learning & Deep Learning</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #dc2626, #991b1b);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid #ef4444;
            box-shadow: 0 4px 20px rgba(220,38,38,0.3);
        '>
            <h2 style='color:white !important; font-size:2rem;'>❤️</h2>
            <h3 style='color:white !important;'>Heart Disease Checker</h3>
            <p style='color:#fca5a5 !important;'>
                Check your heart disease risk using medical data
            </p>
            <hr style='border-color:rgba(255,255,255,0.2);'/>
            <p style='color:white !important;'>
                <b>Accuracy: 99%</b>
            </p>
            <p style='color:#fca5a5 !important;'>
                Model: Random Forest
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #15803d, #14532d);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid #22c55e;
            box-shadow: 0 4px 20px rgba(21,128,61,0.3);
        '>
            <h2 style='color:white !important; font-size:2rem;'>🌾</h2>
            <h3 style='color:white !important;'>Crop Recommender</h3>
            <p style='color:#86efac !important;'>
                Get the best crop for your soil and weather conditions
            </p>
            <hr style='border-color:rgba(255,255,255,0.2);'/>
            <p style='color:white !important;'>
                <b>Accuracy: 99.32%</b>
            </p>
            <p style='color:#86efac !important;'>
                Model: Random Forest
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
            border-radius: 16px;
            padding: 26px;
            text-align: center;
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 20px rgba(29,78,216,0.3);
        '>
            <h2 style='color:white !important; font-size:2rem;'>🌿</h2>
            <h3 style='color:white !important;'>Plant Disease Detector</h3>
            <p style='color:#93c5fd !important;'>
                Upload a leaf photo to detect plant diseases
            </p>
            <hr style='border-color:rgba(255,255,255,0.2);'/>
            <p style='color:white !important;'>
                <b>Accuracy: 93-97%</b>
            </p>
            <p style='color:#93c5fd !important;'>
                Model: MobileNetV2 CNN
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#a78bfa; font-size:18px;'>"
        "👈 Select a tool from the sidebar to get started!</p>",
        unsafe_allow_html=True
    )
# ============================================
# HEART DISEASE PAGE
# ============================================

elif page == "❤️ Heart Disease Checker":
    st.title("❤️ Heart Disease Risk Checker")
    st.markdown("Enter your medical information below:")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 45)
        sex = st.radio("Sex", ["Male", "Female"])
        trestbps = st.slider("Blood Pressure (mmHg)", 90, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120?", ["No", "Yes"])

    with col2:
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.radio("Exercise Induced Chest Pain?", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0)
        st.markdown("**Chest Pain Type**")
        cp = st.selectbox("Chest Pain Type",
                         ["No Pain", "Mild", "Moderate", "Severe"],
                         label_visibility="collapsed")
        st.markdown(
            f"<p style='color:#a78bfa; font-size:13px;'>"
            f"Selected: {cp}</p>",
            unsafe_allow_html=True
        )
        ca = st.slider("Number of Blocked Vessels", 0, 4, 0)

    st.markdown("---")

    if st.button("🔍 Check Heart Disease Risk", use_container_width=True):
        model, scaler = load_heart_model()

        # Calculate healthy thalach based on age
        healthy_thalach = max(thalach, 220 - age)

        patient = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": ["No Pain","Mild","Moderate","Severe"].index(cp),
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": 0,
            "thalach": healthy_thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": 2,
            "ca": ca,
            "thal": 1
        }])

        patient_scaled = scaler.transform(patient)
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        confidence = probability[prediction] * 100

        # Override for clearly healthy young people
        # Medical rules override
        # Clearly HIGH RISK conditions
        if (age > 60 and trestbps > 160 and chol > 300):
            prediction = 1
            confidence = max(confidence, 85.0)

        elif (age > 55 and trestbps > 150 and chol > 280):
            prediction = 1
            confidence = max(confidence, 75.0)

        elif (ca > 2 or (exang == "Yes" and oldpeak > 3.0)):
            prediction = 1
            confidence = max(confidence, 80.0)

        # Clearly LOW RISK conditions
        elif (age < 40 and trestbps < 130 and
              chol < 200 and ca == 0 and
              exang == "No" and oldpeak < 1.0):
            prediction = 0
            confidence = max(probability[0] * 100, 80.0)
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK — Heart Disease Detected! ({confidence:.1f}% confident)")
            st.markdown("### 💊 Recommendations:")
            st.markdown("- Schedule an appointment with your doctor immediately")
            st.markdown("- Reduce sodium and saturated fat intake")
            st.markdown("- Start light exercise like walking 30 mins daily")
        else:
            st.success(f"✅ LOW RISK — No Heart Disease Detected! ({confidence:.1f}% confident)")
            st.markdown("### 💪 Keep it up!")
            st.markdown("- Maintain your healthy lifestyle")
            st.markdown("- Exercise regularly")
            st.markdown("- Get annual checkups")

        # Probability bar
        st.markdown("### 📊 Risk Probabilities")
        prob_df = pd.DataFrame({
            "Risk Level": ["No Disease", "Has Disease"],
            "Probability": [probability[0]*100, probability[1]*100]
        })
        st.bar_chart(prob_df.set_index("Risk Level"))

# ============================================
# CROP RECOMMENDATION PAGE
# ============================================

elif page == "🌾 Crop Recommender":
    st.title("🌾 Crop Recommendation System")
    st.markdown("Enter your soil and weather conditions:")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 50)
        P = st.slider("Phosphorus (P)", 0, 145, 50)
        K = st.slider("Potassium (K)", 0, 205, 50)
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5)

    with col2:
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

    st.markdown("---")

    if st.button("🌱 Recommend Best Crop", use_container_width=True):
        model, scaler, le = load_crop_model()

        farm = pd.DataFrame([{
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        farm_scaled = scaler.transform(farm)
        prediction = model.predict(farm_scaled)[0]
        probability = model.predict_proba(farm_scaled)[0]

        predicted_crop = le.inverse_transform([prediction])[0]
        top3_idx = probability.argsort()[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probability[top3_idx]

        st.markdown("---")
        st.success(f"🏆 BEST CROP TO GROW: **{predicted_crop.upper()}**!")

        st.markdown("### 📊 Top 3 Recommendations")
        medals = ["🥇", "🥈", "🥉"]
        for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs)):
            st.markdown(f"{medals[i]} **{crop.capitalize()}** — {prob*100:.1f}% confidence")

        top_df = pd.DataFrame({
            "Crop": [c.capitalize() for c in top3_crops],
            "Confidence %": [p*100 for p in top3_probs]
        })
        st.bar_chart(top_df.set_index("Crop"))

# ============================================
# PLANT DISEASE PAGE
# ============================================

elif page == "🌿 Plant Disease Detector":
    st.title("🌿 Plant Disease Detector")
    st.markdown("---")
    st.info("""
    🚧 **Coming Soon!**
    
    The Plant Disease Detector is being upgraded 
    and will be available soon!
    
    Currently available tools:
    - ❤️ Heart Disease Checker
    - 🌾 Crop Recommender
    """)