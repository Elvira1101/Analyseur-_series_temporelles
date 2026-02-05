
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.models import predict_prophet, load_model
import os

app = FastAPI(title="Forecast API")

MODEL_PATH = os.path.join("models", "prophet_model.joblib")
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

class ForecastRequest(BaseModel):
    periods: int = 30
    freq: str = "D"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    global model
    if model is None:
        return {"error":"model not found, train first"}
    fc = predict_prophet(model, periods=req.periods, freq=req.freq)
   
    return fc.to_dict(orient='records')
