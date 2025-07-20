import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('models/xgb_model.pkl', 'rb'))

# Load data
df = pd.read_csv('data/heart_cleveland_upload.csv')
X = df.drop('condition', axis=1)

# SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot beeswarm (global explanation)
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("models/global_explanation.png")  # Save as image instead of HTML
plt.show()
