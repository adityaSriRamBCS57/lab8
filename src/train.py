import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

STUDENT_NAME = "Aditya Sri Ram"
ROLL_NO = "2022BCS0057"

data = pd.read_csv("data/housing.csv")
print(f"[{ROLL_NO}] Dataset size: {len(data)} rows")

# Drop rows with missing values
data = data.dropna()

# Encode ocean_proximity if presen
if "ocean_proximity" in data.columns:
    data = pd.get_dummies(data, columns=["ocean_proximity"])

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

print(f"[{ROLL_NO}] RMSE: {rmse:.2f}")
print(f"[{ROLL_NO}] R²:   {r2:.4f}")
print(f"[{ROLL_NO}] Training samples: {len(X_train)}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names
feature_meta = {"features": list(data.drop("median_house_value", axis=1).columns)}
with open("models/feature_meta.json", "w") as f:
    json.dump(feature_meta, f)

metrics = {
    "student_name": STUDENT_NAME,
    "roll_no": ROLL_NO,
    "dataset_size": len(data),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "rmse": round(rmse, 2),
    "r2": round(r2, 4),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
