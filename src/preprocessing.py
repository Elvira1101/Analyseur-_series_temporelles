# src/preprocessing.py
import pandas as pd
import numpy as np
from typing import Optional

# Try to import sklearn scalers; provide lightweight fallback if not available
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

def ensure_freq(df: pd.DataFrame, freq: str='D', ts_col: str='ds', value_col: str='y') -> pd.DataFrame:
    """
    Force une fréquence régulière (asfreq). Agrège les doublons par moyenne avant resampling.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col])

    # Agrégation des doublons par moyenne pour éviter "duplicate labels"
    if df[ts_col].duplicated().any():
        df = df.groupby(ts_col)[value_col].mean().reset_index()

    df = df.set_index(ts_col).sort_index()
    df = df.asfreq(freq)  # remplit les trous (index DatetimeIndex)
    return df.reset_index()

def impute_missing(df: pd.DataFrame, method: str='interpolate') -> pd.DataFrame:
    """
    Impute les valeurs manquantes : 'interpolate' | 'ffill' | 'bfill' | 'zero'
    """
    df = df.copy()
    if method == 'interpolate':
        df['y'] = df['y'].interpolate()
    elif method == 'ffill':
        df['y'] = df['y'].fillna(method='ffill')
    elif method == 'bfill':
        df['y'] = df['y'].fillna(method='bfill')
    elif method == 'zero':
        df['y'] = df['y'].fillna(0)
    else:
        raise ValueError("method must be one of 'interpolate', 'ffill', 'bfill', 'zero'")
    return df

def cap_outliers_iqr(df: pd.DataFrame, col: str='y', multiplier: float=1.5) -> pd.DataFrame:
    """
    Cappe les outliers selon IQR (clip aux bornes).
    """
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    df[col] = df[col].clip(lower=low, upper=high)
    return df

# --- Normalisation avec fallback ---
def _minmax_numpy(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    if np.nanmin(arr) == np.nanmax(arr):
        return pd.Series(np.zeros_like(arr), index=series.index)
    scaled = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    return pd.Series(scaled, index=series.index)

def _standard_numpy(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(arr), index=series.index)
    scaled = (arr - mean) / std
    return pd.Series(scaled, index=series.index)

def normalize(df: pd.DataFrame, method: str='MinMax', col: str='y', scaler_obj: Optional[object]=None) -> pd.DataFrame:
    """
    Normalise la colonne `col`. Utilise sklearn si disponible, sinon fallback numpy/pandas.
    method: 'MinMax' or 'Standard'
    """
    df = df.copy()
    if method not in ('MinMax','Standard'):
        raise ValueError("method must be 'MinMax' or 'Standard'")

    if _HAVE_SKLEARN and scaler_obj is None:
        if method == 'MinMax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
        return df

    # fallback (no sklearn or scaler_obj provided)
    if method == 'MinMax':
        df[col] = _minmax_numpy(df[col])
    else:
        df[col] = _standard_numpy(df[col])
    return df
