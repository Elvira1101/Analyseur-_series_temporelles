# src/anomalies.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def zscore_anomalies(df: pd.DataFrame, col: str='y', threshold: float=3.0) -> pd.DataFrame:
    mean = df[col].mean()
    std = df[col].std()
    df['z'] = (df[col] - mean) / (std if std!=0 else 1)
    df['anomaly_z'] = df['z'].abs() > threshold
    return df

def isolation_forest_anomalies(df: pd.DataFrame, cols: list=['y'], contamination: float=0.01) -> pd.DataFrame:
    clf = IsolationForest(contamination=contamination, random_state=42)
    X = df[cols].fillna(0).values
    preds = clf.fit_predict(X)
    df['anomaly_if'] = preds == -1
    return df
