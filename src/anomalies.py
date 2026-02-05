# src/anomalies.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def zscore_anomalies(df: pd.DataFrame, col: str='y', threshold: float=3.0) -> pd.DataFrame:
    """Compute z-score anomalies safely.

    For small or constant series, returns the DataFrame with 'z' and 'anomaly_z'
    columns set to 0/False to avoid breaking downstream code.
    """
    df = df.copy()
    # ensure numeric
    s = pd.to_numeric(df[col], errors='coerce')
    valid = s.dropna()
    if len(valid) < 2 or valid.std() == 0 or pd.isna(valid.std()):
        df['z'] = 0.0
        df['anomaly_z'] = False
        return df

    mean = valid.mean()
    std = valid.std()
    df['z'] = (s - mean) / (std if std != 0 else 1.0)
    df['anomaly_z'] = df['z'].abs() > threshold
   
    df['anomaly_z'] = df['anomaly_z'].fillna(False).astype(bool)
    return df

def isolation_forest_anomalies(df: pd.DataFrame, cols: list=['y'], contamination: float=0.01) -> pd.DataFrame:
    """Isolation Forest based anomalies with safety checks."""
    df = df.copy()
   
    if df.shape[0] < 5:
        df['anomaly_if'] = False
        return df
    X = df[cols].fillna(0).values
    
    if (X.max(axis=0) - X.min(axis=0) == 0).all():
        df['anomaly_if'] = False
        return df
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)
    df['anomaly_if'] = preds == -1
    return df
