from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import json
import numpy as np
import os

STUDENT_NAME = "Aditya Sri Ram"
ROLL_NO = "2022BCS0057"

app = FastAPI(title="California Housing API", version="1.0.0")

model = None
scaler = None
feature_meta = None

@app.on_event("startup")
def load_model():
    global model, scaler, feature_meta
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_meta.json") as f:
        feature_meta = json.load(f)

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "model": "GradientBoostingRegressor",
    }

class PredictRequest(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {
        "predicted_house_value": round(float(pred), 2),
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
    }
