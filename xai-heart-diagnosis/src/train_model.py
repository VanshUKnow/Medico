import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#For loading the data
df = pd.read_csv('data/heart_cleveland_upload.csv')

#Replacing 'target' with 'condition' (because the end result in dataset that we used is target. Changing it to condition will be better decision)
X = df.drop('condition', axis=1)
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

#Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

#Save model
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
