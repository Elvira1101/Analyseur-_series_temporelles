# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import ensure_freq, impute_missing, cap_outliers_iqr

def test_ensure_freq_and_impute():
    df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=3, freq='D'), 'y':[1, None, 3]})
    df2 = ensure_freq(df, freq='D')
    assert 'ds' in df2.columns
    df3 = impute_missing(df2, method='interpolate')
    assert df3['y'].isna().sum() == 0

def test_cap_outliers_iqr():
    df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=5, freq='D'), 'y':[1,2,3,1000,2]})
    df2 = cap_outliers_iqr(df.copy(), col='y')
    assert df2['y'].max() < 1000
