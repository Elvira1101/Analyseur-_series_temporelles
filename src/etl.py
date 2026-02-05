# src/etl.py
import pandas as pd
from typing import Optional, Union
import io
from io import BytesIO


def _read_bytes(source):
    """Return a BytesIO for file-like sources or the original path for string paths."""
    if hasattr(source, 'read'):
        data = source.read()
        try:
            source.seek(0)
        except Exception:
            pass
        return BytesIO(data)
    return source


import csv

def read_raw_csv(source: Union[str, BytesIO], seps=None, encodings=None) -> pd.DataFrame:
    """Try multiple separators and encodings to read a CSV and return a DataFrame preview.
    Adds delimiter sniffing using csv.Sniffer when possible.
    Works with file paths or file-like objects (e.g., Streamlit's UploadedFile).
    """
    seps = seps or [',', ';', '\t']
    encodings = encodings or ['utf-8', 'latin-1']

    def try_read(buf, sep, enc):
        try:
            return pd.read_csv(buf, sep=sep, encoding=enc)
        except Exception:
            return None

    # file-like: we have to reuse the raw bytes for each attempt
    if hasattr(source, 'read'):
        raw = source.read()
        try:
            source.seek(0)
        except Exception:
            pass

        # Try delimiter sniffing on a decoded sample
        for enc in encodings:
            try:
                sample = raw.decode(enc, errors='ignore')[:8192]
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                sep = dialect.delimiter
                buf = BytesIO(raw)
                df = try_read(buf, sep, enc)
                if df is not None and df.shape[1] > 1:
                    return df
            except Exception:
                continue

        # fallback to trying known separators and encodings
        for enc in encodings:
            for sep in seps:
                try:
                    buf = BytesIO(raw)
                    df = try_read(buf, sep, enc)
                    if df is not None and (df.shape[1] > 1 or sep == ','):
                        return df
                except Exception:
                    continue
        # final fallback
        try:
            return pd.read_csv(BytesIO(raw))
        except Exception:
            return pd.DataFrame()
    else:
        # path-like: try sniffing on file start
        for enc in encodings:
            try:
                with open(source, 'r', encoding=enc, errors='ignore') as fh:
                    sample = fh.read(8192)
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    sep = dialect.delimiter
                    df = try_read(source, sep, enc)
                    if df is not None and df.shape[1] > 1:
                        return df
            except Exception:
                continue
        # path fallback
        for enc in encodings:
            for sep in seps:
                try:
                    df = try_read(source, sep, enc)
                    if df is not None and df.shape[1] > 1:
                        return df
                except Exception:
                    continue
        try:
            return pd.read_csv(source)
        except Exception:
            return pd.DataFrame()


def _clean_numeric_column(s: pd.Series) -> pd.Series:
    """Standardize a column by replacing ',' with '.' if it looks like a French decimal."""
    if s.dtype == object:
        sample = s.dropna().astype(str).head(50)
        if sample.str.match(r"^-?\d+[,]\d+$").sum() > 0:
            return s.astype(str).str.replace(',', '.').replace({'': None})
    return s


def auto_fix_time_series(df: pd.DataFrame, ts_col: Optional[str] = None, value_col: Optional[str] = None) -> pd.DataFrame:
    """Attempt an automated fix for common data issues like incorrect column mapping or format."""
    tmp = df.copy()
    for col in tmp.select_dtypes(include=['object']).columns:
        tmp[col] = _clean_numeric_column(tmp[col])
    try:
        return prepare_time_series(tmp, ts_col=ts_col, value_col=value_col)
    except Exception:
        # If preparation fails, try fallback mapping
        numeric_cols = tmp.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            # try forcing numeric on all columns
            for col in tmp.columns:
                try:
                    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
                except Exception:
                    pass
            numeric_cols = tmp.select_dtypes(include='number').columns.tolist()

        if numeric_cols:
            best_y = numeric_cols[0]
            # find a date column or create one
            return prepare_time_series(tmp, value_col=best_y)
        
        raise ValueError("Impossible d'auto-corriger les données : aucune colonne numérique trouvée.")


def prepare_time_series(df: pd.DataFrame, ts_col: Optional[str] = None, value_col: Optional[str] = None, tz: Optional[str] = None) -> pd.DataFrame:
    """Normalize a DataFrame to the expected time series format with columns `ds` (datetime) and `y` (numeric)."""
    df = df.copy()

    # cleanup header whitespace & BOM
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    # Process potential French decimals before any type conversion
    if value_col and value_col in df.columns:
        df[value_col] = _clean_numeric_column(df[value_col])
    else:
        for col in df.select_dtypes(include=['object']).columns:
             df[col] = _clean_numeric_column(df[col])

    cols_lower = [c.lower() for c in df.columns]

    # detect or use provided date column
    date_candidates = ['date', 'ds', 'timestamp', 'time']
    date_col = None
    if ts_col:
        matches = [c for c in df.columns if c.lower() == ts_col.lower()]
        if matches:
            date_col = matches[0]
    
    if date_col is None:
        date_col = next((df.columns[i] for i, c in enumerate(cols_lower) if c in date_candidates), None)

    if date_col is not None:
        df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df['ds'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='D')

    # detect or use value column
    if value_col:
        matches = [c for c in df.columns if c.lower() == value_col.lower()]
        val_col = matches[0] if matches else None
    else:
        val_candidates = ['inventory', 'stock', 'sales', 'value', 'y', 'units', 'balance', 'price', 'amount']
        val_col = next((df.columns[i] for i, c in enumerate(cols_lower) if any(cand in c for cand in val_candidates)), None)

    if val_col is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            val_col = numeric_cols[0]
        else:
            raise ValueError(f"Aucune colonne valeur détectée; colonnes présentes: {list(df.columns)}")

    df['y'] = pd.to_numeric(df[val_col], errors='coerce')

    if tz:
        if df['ds'].dt.tz is None:
            df['ds'] = df['ds'].dt.tz_localize(tz)

    df = df[['ds', 'y']].dropna(subset=['y']).reset_index(drop=True)
    return df


def load_csv(source: Union[str, BytesIO], ts_col: str = None, value_col: str = None, tz: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV (with robust detection) and prepare it as a timeseries DataFrame (ds, y)."""
    raw = read_raw_csv(source)
    return prepare_time_series(raw, ts_col=ts_col, value_col=value_col, tz=tz)
