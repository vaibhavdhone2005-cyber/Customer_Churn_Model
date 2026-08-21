import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. PAGE CONFIGURATION & EXECUTIVE THEME
# ==========================================
st.set_page_config(
    page_title="Executive Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Business Meeting UI
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .header-box h1 {
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0.5rem;
        color: #F8FAFC;
    }
    .header-box p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Input Section Cards */
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Result Metric Badges */
    .status-badge-high {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .status-badge-low {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .status-badge-med {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Predict Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }
</style>
""", unsafe_allow_dict_style=True)


# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    model_path = "customer_churn_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"❌ Model file `{model_path}` not found. Please upload it to the working directory.")
        return None

model = load_model()

# ==========================================
# 3. HEADER & PRESENTATION PRESETS
# ==========================================
st.markdown("""
<div class="header-box">
    <h1>💼 Customer Churn Intelligence Dashboard</h1>
    <p>Predict customer attrition risk in real-time and drive proactive retention strategies.</p>
</div>
""", unsafe_allow_dict_style=True)

# Sidebar Preset Scenarios for live meeting speed
st.sidebar.image("https://img.icons8.com/color/96/dashboard-layout.png", width=60)
st.sidebar.title("⚡ Demo Controls")
st.sidebar.write("Load preset profiles during meetings for instant demonstration:")

preset = st.sidebar.radio(
    "Choose Preset Scenario:",
    ["Custom Input", "🟢 Low Risk (Loyal Enterprise)", "🔴 High Risk (At-Risk Customer)"]
)

# Set Default Values based on Selection
if preset == "🟢 Low Risk (Loyal Enterprise)":
    defaults = {"age": 42, "gender": "Male", "tenure": 36, "usage": 22, "calls": 1, "delay": 2, "sub": "Premium", "contract": "Annual", "spend": 4500, "interaction": 5}
elif preset == "🔴 High Risk (At-Risk Customer)":
    defaults = {"age": 28, "gender": "Female", "tenure": 4, "usage": 3, "calls": 8, "delay": 18, "sub": "Basic", "contract": "Monthly", "spend": 350, "interaction": 25}
else:
    defaults = {"age": 35, "gender": "Male", "tenure": 12, "usage": 10, "calls": 3, "delay": 5, "sub": "Standard", "contract": "Quarterly", "spend": 1200, "interaction": 10}

# Feature Categorical Mappings (Standard Ordinal Encoding)
GENDER_MAP = {"Female": 0, "Male": 1}
SUB_MAP = {"Basic": 0, "Standard": 1, "Premium": 2}
CONTRACT_MAP = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

# ==========================================
# 4. INPUT FORM LAYOUT
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><div class="card-title">👤 Customer Demographics & Subscription</div>', unsafe_allow_dict_style=True)
    
    age = st.number_input("Age", min_value=18, max_value=100, value=defaults["age"])
    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
    sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], index=["Basic", "Standard", "Premium"].index(defaults["sub"]))
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"], index=["Monthly", "Quarterly", "Annual"].index(defaults["contract"]))
    total_spend = st.number_input("Total Spend ($)", min_value=0, max_value=100000, value=defaults["spend"], step=100)
    
    st.markdown('</div>', unsafe_allow_dict_style=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">📈 Engagement & Interaction Metrics</div>', unsafe_allow_dict_style=True)
    
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=defaults["tenure"])
    usage_freq = st.slider("Usage Frequency (Days / Month)", min_value=0, max_value=30, value=defaults["usage"])
    support_calls = st.slider("Support Calls Received", min_value=0, max_value=20, value=defaults["calls"])
    payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=60, value=defaults["delay"])
    last_interaction = st.number_input("Days Since Last Interaction", min_value=0, max_value=60, value=defaults["interaction"])
    
    st.markdown('</div>', unsafe_allow_dict_style=True)

# ==========================================
# 5. PREDICTION & ANALYTICS OUTPUT
# ==========================================
st.markdown("<br>", unsafe_allow_dict_style=True)
predict_clicked = st.button("🔍 Evaluate Customer Churn Probability")

if predict_clicked or preset != "Custom Input":
    if model is not None:
        # Prepare Feature Vector according to model requirements
        feature_values = [
            age,
            GENDER_MAP[gender],
            tenure,
            usage_freq,
            support_calls,
            payment_delay,
            SUB_MAP[sub_type],
            CONTRACT_MAP[contract_length],
            total_spend,
            last_interaction
        ]
        
        feature_names = [
            "Age", "Gender", "Tenure", "Usage Frequency",
            "Support Calls", "Payment Delay", "Subscription Type",
            "Contract Length", "Total Spend", "Last Interaction"
        ]
        
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # Inference
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        churn_prob = probabilities[1] * 100
        
        st.markdown("---")
        st.subheader("📊 Executive Analysis & Decision Support")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if churn_prob >= 60:
                st.markdown(f"""
                <div class="status-badge-high">
                    <h3 style="margin:0; font-size: 1.2rem;">CRITICAL RISK LEVEL</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 3rem;">{churn_prob:.1f}%</h1>
                    <p style="margin:0;">High Risk of Customer Attrition</p>
                </div>
                """, unsafe_allow_dict_style=True)
            elif churn_prob >= 30:
                st.markdown(f"""
                <div class="status-badge-med">
                    <h3 style="margin:0; font-size: 1.2rem;">MODERATE RISK LEVEL</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 3rem;">{churn_prob:.1f}%</h1>
                    <p style="margin:0;">Monitor Engagement & Solicit Feedback</p>
                </div>
                """, unsafe_allow_dict_style=True)
            else:
                st.markdown(f"""
                <div class="status-badge-low">
                    <h3 style="margin:0; font-size: 1.2rem;">LOW RISK LEVEL</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 3rem;">{churn_prob:.1f}%</h1>
                    <p style="margin:0;">Customer is Stable & Highly Retained</p>
                </div>
                """, unsafe_allow_dict_style=True)

        with res_col2:
            st.markdown("#### Recommended Strategic Action")
            if churn_prob >= 60:
                st.error("🚨 **Immediate Intervention Required**")
                st.write("* **Action Item 1:** Assign a dedicated Account Executive within 24 hours.")
                st.write("* **Action Item 2:** Offer a 15% promotional extension discount on annual renewal.")
                st.write("* **Action Item 3:** Resolve open support queries immediately.")
            elif churn_prob >= 30:
                st.warning("⚠️ **Proactive Engagement Suggested**")
                st.write("* **Action Item 1:** Send automated product feature tips and check-in survey.")
                st.write("* **Action Item 2:** Review support call history to identify key friction points.")
            else:
                st.success("✅ **Healthy Account Status**")
                st.write("* **Action Item 1:** Target for upsell or premium feature add-on.")
                st.write("* **Action Item 2:** Invite to join customer advocacy / loyalty advisory panel.")
