# src/features.py
import pandas as pd

def add_lags(df: pd.DataFrame, col: str='y', lags: list=[1,7,14]) -> pd.DataFrame:
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling(df: pd.DataFrame, col: str='y', windows: list=[3,7,14]) -> pd.DataFrame:
    for w in windows:
        df[f'{col}_rmean_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
    return df
