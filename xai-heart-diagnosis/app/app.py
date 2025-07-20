import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import sys
import pickle
import google.generativeai as genai

#For loading gemini function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_explainer import explain_for_doctors

#Environment (loading .env file where our API key is saved)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#Gemini model
model_gemini = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")

#For loading model
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.Explainer(model)

#Styling
st.set_page_config(page_title="Heart Disease Diagnosis", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #fafafa;
    }
    .main {
        padding: 1.5rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #ff6f00;
    }
    .stButton>button {
        background-color: #ff6f00;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e65c00;
    }
    .stSidebar {
        background-color: #fff6e6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Diagnosis with Explainable AI")
st.markdown("Use patient health metrics to predict heart disease and understand the reasons using AI.")

# Sidebar input
st.sidebar.header("Enter Patient Data")
if "history" not in st.session_state:
    st.session_state["history"] = []

age = st.sidebar.slider("Age", 10, 100, 50)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.slider("Chest Pain Type (0–3)", 0, 3, 0)
trestbps = st.sidebar.slider("Resting BP", 60, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting Sugar >120", [0, 1])
restecg = st.sidebar.slider("Resting ECG", 0, 2, 1)
thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.slider("ST Slope", 0, 2, 1)
ca = st.sidebar.slider("Major Vessels (0–4)", 0, 4, 0)
thal = st.sidebar.slider("Thalassemia (0–3)", 0, 3, 2)

input_data = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
features = list(input_data.keys())
input_df = pd.DataFrame([input_data])

st.subheader("Input Features")
st.dataframe(input_df.style.highlight_max(axis=1), use_container_width=True)

#Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    result_label = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected"

    st.session_state["history"].append({
        "inputs": input_data,
        "label": result_label,
        "confidence": f"{prediction_proba:.2%}"
    })

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"{result_label}")
    else:
        st.success(f"{result_label}")
    st.info(f"Confidence Score: {prediction_proba:.2%}")

    # SHAP explanation
    shap_values = explainer(input_df)
    shap_df = pd.DataFrame({
        'feature': features,
        'shap_value': shap_values[0].values
    }).sort_values(by='shap_value', key=abs, ascending=True)

    st.subheader("Feature Impact (SHAP)")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff6f00' if v > 0 else '#1f77b4' for v in shap_df['shap_value']]
    ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
    ax.set_xlabel("Impact")
    st.pyplot(fig)

    #Gemini explanation
    if model_gemini:
        st.subheader("Gemini Explanation for Doctors")
        top_features = shap_df.tail(5)['feature'].tolist()

        try:
            explanation = explain_for_doctors(
                input_features=input_data,
                prediction_label=result_label,
                shap_values=shap_values[0].values.tolist(),
                top_features=top_features
            )
            st.success("Explanation generated")
            st.markdown(f"{explanation}")
            st.download_button(
                "Download Explanation",
                data=explanation,
                file_name="gemini_explanation.txt"
            )
        except Exception as e:
            st.error(f"Gemini explanation failed: {e}")

#Sidebar history (until the page is refreshed)
st.sidebar.header("History & Export")
if st.sidebar.checkbox("Show Prediction History"):
    for i, item in enumerate(st.session_state["history"][::-1], 1):
        st.sidebar.markdown(f"**{i}.** {item['label']} – {item['confidence']}")
        with st.sidebar.expander("Details"):
            st.sidebar.json(item["inputs"])

if st.sidebar.button("Download History as CSV") and st.session_state["history"]:
    df_history = pd.DataFrame(st.session_state["history"])
    st.sidebar.download_button(
        "Download CSV",
        data=df_history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )
