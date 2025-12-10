# src/etl.py
import pandas as pd
from typing import Optional, Union
import io

def load_csv(source: Union[str, "io.BytesIO"], ts_col: str = 'Date', value_col: str = None, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Charge un CSV depuis un chemin ou un UploadedFile (streamlit) et renvoie un DataFrame.
    - source: chemin ou file-like
    - ts_col: nom probable de la colonne date (sera cherché case-insensitive)
    - value_col: nom probable de la colonne valeur (optionnel)
    """
    # utilise pandas pour lire la source (chemin ou buffer)
    df = pd.read_csv(source)

    # cleanup header whitespace & BOM
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # normalisation temporaire des noms pour detection
    cols_lower = [c.lower() for c in df.columns]

    # détecte colonne date si différente
    date_candidates = ['date','ds','timestamp','time']
    date_col = next((df.columns[i] for i,c in enumerate(cols_lower) if c in date_candidates), None)
    if date_col is None:
        raise ValueError(f"Aucune colonne date détectée parmi {date_candidates}; colonnes présentes: {list(df.columns)}")

    # détecte colonne valeur
    if value_col is None:
        val_candidates = ['inventory level','inventory_level','inventory','stock','sales','value','y','units sold','units_sold']
        value_col = next((df.columns[i] for i,c in enumerate(cols_lower) if c in val_candidates), None)
        if value_col is None:
            raise ValueError(f"Aucune colonne valeur détectée parmi {val_candidates}; colonnes présentes: {list(df.columns)}")

    # rename to standard names used in pipeline: ds, y
    df = df.rename(columns={date_col:'ds', value_col:'y'})

    # convert ds to datetime
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    if tz:
        if df['ds'].dt.tz is None:
            df['ds'] = df['ds'].dt.tz_localize(tz)

    return df
