import pandas as pd
import pytest
from src.anomalies import zscore_anomalies, isolation_forest_anomalies
from src.models import train_prophet


def test_zscore_small_series():
    df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=1), 'y':[1]})
    out = zscore_anomalies(df, col='y')
    assert 'anomaly_z' in out.columns
    assert out['anomaly_z'].dtype == bool
    assert out['anomaly_z'].sum() == 0


def test_isolation_forest_small():
    df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=3), 'y':[1,2,3]})
    out = isolation_forest_anomalies(df, cols=['y'])
    assert 'anomaly_if' in out.columns
    assert out['anomaly_if'].sum() == 0


def test_train_prophet_constant_y():
    df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=5), 'y':[5,5,5,5,5]})
    with pytest.raises(ValueError):
        train_prophet(df)
