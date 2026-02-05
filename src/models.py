# src/models.py
from prophet import Prophet # pyright: ignore[reportMissingImports]
import joblib, os
import pandas as pd
from typing import Tuple

MODELS_DIR = "models"

def train_prophet(df: pd.DataFrame, save_name: str='prophet_model.joblib') -> Tuple[Prophet, str]:
    # Basic validation to provide clearer error messages
    if not isinstance(df, pd.DataFrame):
        raise ValueError("train_prophet: attendu un DataFrame pandas.")
    required = ['ds', 'y']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"train_prophet: la colonne requise '{c}' est absente du DataFrame.")
    clean = df.dropna(subset=['ds','y'])
    if len(clean) < 2:
        raise ValueError("train_prophet: le DataFrame contient moins de 2 lignes valides (non-NaN) pour l'entraînement.")
    # optional: ensure sufficient variability
    if clean['y'].nunique() <= 1:
        raise ValueError("train_prophet: la colonne 'y' semble constante — impossible d'entraîner un modèle de prévision utile.")

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(clean)
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
    """Return decomposition components for the provided dataframe.
    If the dataframe is empty or Prophet raises an error (e.g., no rows), return an empty DataFrame
    with the expected columns (if any).
    """
    # if df is not a DataFrame or has no rows, return empty with expected columns
    expected_cols = ['ds','trend','weekly','yearly']
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        return pd.DataFrame(columns=expected_cols)

    try:
        future = model.predict(df)
    except ValueError as e:
        # Prophet may raise ValueError('Dataframe has no rows.') — return empty result
        if 'no rows' in str(e).lower():
            return pd.DataFrame(columns=expected_cols)
        # re-raise unknown errors
        raise

    # some models provide weekly/yearly; select safely
    cols = [c for c in expected_cols if c in future.columns]
    return future[cols]
