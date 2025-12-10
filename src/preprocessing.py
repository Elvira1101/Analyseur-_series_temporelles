# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional

def ensure_freq(df, freq='D', ts_col='ds', value_col='y'):
    """
    Fixe la fréquence en résolvant les doublons avant resampling.
    """
    import pandas as pd

    # Conversion en datetime
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    # Suppression des lignes où la date est invalide
    df = df.dropna(subset=[ts_col])

    # ----- 🔥 IMPORTANT : supprimer ou agréger les doublons -----
    if df[ts_col].duplicated().any():
        # Option : moyenne si plusieurs valeurs ce jour-là
        df = df.groupby(ts_col)[value_col].mean().reset_index()

    # Index pour resampling
    df = df.set_index(ts_col).sort_index()

    # Resample (complète les trous)
    df = df.asfreq(freq)

    return df.reset_index()


def impute_missing(df: pd.DataFrame, method: str='interpolate') -> pd.DataFrame:
    """Impute missing values in y"""
    if method == 'interpolate':
        df['y'] = df['y'].interpolate()
    elif method == 'ffill':
        df['y'] = df['y'].fillna(method='ffill')
    elif method == 'bfill':
        df['y'] = df['y'].fillna(method='bfill')
    elif method == 'zero':
        df['y'] = df['y'].fillna(0)
    else:
        raise ValueError("method must be interpolate|ffill|bfill|zero")
    return df

def cap_outliers_iqr(df: pd.DataFrame, col: str='y', multiplier: float=1.5) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - multiplier*iqr
    high = q3 + multiplier*iqr
    df[col] = df[col].clip(lower=low, upper=high)
    return df

def normalize(df: pd.DataFrame, method: str='MinMax', col: str='y', scaler_obj: Optional[object]=None):
    if method == 'MinMax':
        scaler = scaler_obj if scaler_obj is not None else MinMaxScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    elif method == 'Standard':
        scaler = scaler_obj if scaler_obj is not None else StandardScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    return df
