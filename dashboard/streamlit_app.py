# dashboard/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.etl import load_csv
from src.preprocessing import ensure_freq, impute_missing, cap_outliers_iqr, normalize
from src.anomalies import zscore_anomalies
from src.models import train_prophet, predict_prophet, decompose_prophet, load_model
import joblib
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Analyseur Séries Temporelles - Retail Store Inventory")

# Upload or default file
uploaded = st.file_uploader("Upload CSV (Date + Inventory Level or ds + y + Category)", type=['csv'])
default_csv = os.path.join("data","retail_store_inventory.csv")

try:
    if uploaded:
        df = load_csv(uploaded)
    else:
        st.info(f"Chargement du fichier test {default_csv}")
        df = load_csv(default_csv)
except Exception as e:
    st.error(f"Erreur chargement CSV : {e}")
    st.stop()

# show columns
st.write("Colonnes détectées :", df.columns.tolist())

# If category exists, show selector before processing
if 'category' in [c.lower() for c in df.columns]:
    # normalize column names lower-case for selecting
    df.columns = df.columns.str.strip()
    cat_col = next((c for c in df.columns if c.lower()=='category'), None)
    if cat_col:
        df[cat_col] = df[cat_col].astype(str).str.strip()
        cats = df[cat_col].unique().tolist()
        sel = st.selectbox("Filtrer par catégorie (optionnel)", ["__all__"] + cats)
        if sel != "__all__":
            df = df[df[cat_col]==sel]

# Ensure 'ds' and 'y' exist (load_csv already tries, but double-check)
cols_lower = [c.lower() for c in df.columns]
if 'ds' not in cols_lower or 'y' not in cols_lower:
    st.error("Les colonnes 'ds' et 'y' sont nécessaires après chargement. Vérifie ton CSV ou le load_csv.")
    st.write("Colonnes actuelles :", df.columns.tolist())
    st.stop()

# normalize column names to standard 'ds' and 'y' exactly
rename_map = {}
for c in df.columns:
    if c.lower() == 'ds':
        rename_map[c] = 'ds'
    if c.lower() == 'y':
        rename_map[c] = 'y'
df = df.rename(columns=rename_map)

# PREPROCESSING UI
st.subheader("Prétraitement & Normalisation")
col1, col2, col3 = st.columns(3)
with col1:
    freq = st.selectbox("Fréquence (resample)", ['D','W','M'], index=0)
with col2:
    impute_method = st.selectbox("Imputation", ['interpolate','ffill','bfill','zero'])
with col3:
    norm_method = st.selectbox("Normalisation", ['Aucune','MinMax','Standard'])

missing_before = int(df['y'].isna().sum())
df = ensure_freq(df, freq=freq)
df = impute_missing(df, method=impute_method)
df = cap_outliers_iqr(df, col='y')
if norm_method != 'Aucune':
    df = normalize(df, method=norm_method)

st.write(f"Valeurs manquantes avant imputation: {missing_before} | après: {int(df['y'].isna().sum())}")

# Plot cleaned series
st.subheader("Série après nettoyage")
st.line_chart(df.set_index('ds')['y'])

# Anomalies
st.subheader("Détection d'anomalies")
df = zscore_anomalies(df, col='y', threshold=3)
st.write(f"Anomalies (z-score >3): {int(df['anomaly_z'].sum())}")
# interactive plotly: series + anomalies
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='y'))
fig.add_trace(go.Scatter(x=df[df['anomaly_z']]['ds'], y=df[df['anomaly_z']]['y'],
                         mode='markers', name='anomalies', marker=dict(color='red', size=8)))
st.plotly_chart(fig, use_container_width=True)

# Forecast
st.subheader("Forecast")
if st.button("Entraîner Prophet"):
    model, path = train_prophet(df[['ds','y']], save_name='prophet_model.joblib')
    st.success(f"Modèle entraîné et sauvegardé : {path}")

model_path = os.path.join("models","prophet_model.joblib")
if os.path.exists(model_path):
    model = load_model(model_path)
    periods = st.number_input("Périodes à prévoir", min_value=1, max_value=365, value=30)
    forecast = predict_prophet(model, periods=periods, freq=freq)
    # plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='historique'))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='forecast'))
    st.plotly_chart(fig2, use_container_width=True)
    st.write(forecast.tail())
    # decomposition
    comps = decompose_prophet(model, df)
    if not comps.empty:
        st.subheader("Décomposition")
        st.line_chart(comps.set_index('ds'))
