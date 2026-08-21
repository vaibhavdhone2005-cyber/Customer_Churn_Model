import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration for Business Presentation
st.set_page_config(
    page_title="Customer Retention Analytics | Executive Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Slate Blue & Emerald Theme)
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .css-1d3b10b, .stSidebar {
        background-color: #1e293b !important;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #065f46 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open('customer_churn_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Banner
st.title("💼 Enterprise Customer Churn Predictor")
st.caption("AI-Powered Risk Assessment Engine for Boardroom Analytics")
st.markdown("---")

# Input Section (Sidebar Layout)
st.sidebar.header("🎯 Customer Profile Inputs")

age = st.sidebar.slider("Age", 18, 100, 35)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=120, value=24)
usage_freq = st.sidebar.slider("Usage Frequency (Monthly Logins)", 1, 30, 12)
support_calls = st.sidebar.slider("Support Calls Logged", 0, 10, 2)
payment_delay = st.sidebar.slider("Payment Delay (Days)", 0, 30, 3)
sub_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.sidebar.number_input("Total Spend ($)", min_value=0, max_value=10000, value=1500)
last_interaction = st.sidebar.slider("Days Since Last Interaction", 0, 30, 5)

# Convert Categoricals to Encoded Numerical Data
gender_encoded = 1 if gender == "Male" else 0
sub_type_map = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

input_features = np.array([[
    age, gender_encoded, tenure, usage_freq, support_calls,
    payment_delay, sub_type_map[sub_type], contract_map[contract_length],
    total_spend, last_interaction
]])

# Main Presentation Dashboard Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Profile Overview")
    profile_df = pd.DataFrame({
        "Attribute": ["Customer Segment", "Total Spend", "Support Engagement", "Contract Term"],
        "Value": [sub_type, f"${total_spend:,.2f}", f"{support_calls} Calls", contract_length]
    })
    st.table(profile_df)
    
    predict_btn = st.button("🚀 Evaluate Churn Risk")

with col2:
    st.subheader("📊 Executive Summary")
    if predict_btn:
        prediction = model.predict(input_features)[0]
        prediction_prob = model.predict_proba(input_features)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        churn_risk = prediction_prob[1] * 100 if hasattr(model, "predict_proba") else (100 if prediction == 1 else 0)

        if prediction == 1 or churn_risk > 50:
            st.markdown(f"""
                <div class="risk-high">
                    <h2 style='margin:0;'>⚠️ High Risk of Churn</h2>
                    <h1 style='margin:10px 0; font-size: 48px;'>{churn_risk:.1f}%</h1>
                    <p style='margin:0;'>Immediate Retention Action Required</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="risk-low">
                    <h2 style='margin:0;'>✅ Account Stable</h2>
                    <h1 style='margin:10px 0; font-size: 48px;'>{100 - churn_risk:.1f}%</h1>
                    <p style='margin:0;'>Retention Confidence Score</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("**Key Action Items:**")
        if prediction == 1 or churn_risk > 50:
            st.error("• Schedule priority check-in with Account Manager.")
            st.error("• Offer contract upgrade discount or tailored loyalty package.")
        else:
            st.success("• Account is healthy. Target for potential cross-sell/up-sell opportunities.")
    else:
        st.info("Adjust values in the left panel and click **Evaluate Churn Risk** to generate real-time predictive insights.")
