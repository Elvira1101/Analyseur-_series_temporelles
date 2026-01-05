# src/models.py
from prophet import Prophet # pyright: ignore[reportMissingImports]
import joblib, os
import pandas as pd
from typing import Tuple

MODELS_DIR = "models"

def train_prophet(df: pd.DataFrame, save_name: str='prophet_model.joblib') -> Tuple[Prophet, str]:
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, save_name)
    joblib.dump(model, path)
    return model, path

def predict_prophet(model, periods: int=30, freq: str='D') -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

def load_model(path: str):
    return joblib.load(path)

def decompose_prophet(model, df: pd.DataFrame):
    future = model.predict(df)
    # some models provide weekly/yearly; select safely
    cols = [c for c in ['ds','trend','weekly','yearly'] if c in future.columns]
    return future[cols]
