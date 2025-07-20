import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

def explain_for_doctors(input_features, prediction_label, shap_values, top_features):
    if not model:
        return "Gemini API key is not configured properly."

    # Build top feature breakdown
    top_feature_details = "\n".join(
        [f"- {feat}: {input_features.get(feat, 'N/A')}" for feat in top_features]
    )

    prompt = f"""
You are an expert medical AI assistant helping doctors understand ML predictions.

Model Output:
The model predicted: **'{prediction_label.upper()}'**

Key Factors Behind This Prediction (Top SHAP Features):
{top_feature_details}

Your Task:
Explain in clear, doctor-friendly terms why this prediction was made.
- Mention what features most strongly influenced the result.
- Don't use technical ML jargon like "model weights", "SHAP plots", etc.
- Make it sound like a doctor is reading a clinical reasoning note.

Avoid code, be concise, clear, and medically intuitive.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini explanation failed: {str(e)}"
