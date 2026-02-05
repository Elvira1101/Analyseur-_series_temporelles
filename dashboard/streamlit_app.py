# dashboard/streamlit_app.py
import sys
import os
import pandas as pd
import streamlit as st
import importlib

# Setup path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import ETL and other modules
from src.etl import read_raw_csv, prepare_time_series, auto_fix_time_series
from src.preprocessing import ensure_freq, impute_missing, cap_outliers_iqr, normalize
from src.anomalies import zscore_anomalies
from src.models import train_prophet, predict_prophet, decompose_prophet, load_model

# --- Helper Functions for Plots ---
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

def plot_anomalies(df, key=None):
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Valeur'))
        if 'anomaly_z' in df.columns:
            anom = df[df['anomaly_z']]
            fig.add_trace(go.Scatter(x=anom['ds'], y=anom['y'],
                                     mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], label='Valeur')
        if 'anomaly_z' in df.columns:
            anom = df[df['anomaly_z']]
            ax.scatter(anom['ds'], anom['y'], color='red', label='Anomalies')
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)

def plot_forecast(df, forecast, key=None):
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historique'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prévision'))
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], label='Historique')
        ax.plot(forecast['ds'], forecast['yhat'], label='Prévision')
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)

# --- Configuration ---
st.set_page_config(
    page_title="Analyseur Séries Temporelles",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
:root{
  --bg: #ffffff;
  --card: rgba(255,255,255,0.03);
  --muted: #9aa4b2;
  --accent-start: #00c2a8;
  --accent-end: #006bff;
}
html, body, [class*="css"]  {
  background: linear-gradient(180deg,var(--bg),#071d28) !important;
  font-family: 'Poppins', sans-serif !important;
  color: #e6eef6;
}
.topbar{display:flex;align-items:center;gap:16px;padding:14px;border-radius:12px;margin-bottom:18px}
.logo{background:linear-gradient(135deg,var(--accent-start),var(--accent-end));color:white;padding:12px;border-radius:10px;font-weight:700;font-size:18px}
.title{font-size:20px;color:#fff;font-weight:700}
.subtitle{color:var(--muted);font-size:13px;margin-top:3px}
.stButton>button{background: linear-gradient(90deg,var(--accent-start),var(--accent-end));border:none;color:white;padding:8px 12px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar'><div class='logo'>TS</div><div><div class='title'>Analyseur Séries Temporelles</div><div class='subtitle'>Analyse & prévision — Retail / toute série temporelle</div></div></div>", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'raw_df' not in st.session_state: st.session_state['raw_df'] = None
if 'df' not in st.session_state: st.session_state['df'] = None
if 'forecast' not in st.session_state: st.session_state['forecast'] = None

# --- Sidebar ---
sidebar = st.sidebar
sidebar.header("1) Importer les données")
uploaded = sidebar.file_uploader("Fichier CSV", type=['csv'])
sep_choice = sidebar.selectbox("Séparateur", ["Auto", ",", ";", "\t"], index=0)
load_example = sidebar.button("Charger un exemple")

if uploaded:
    seps = None if sep_choice == "Auto" else [sep_choice]
    st.session_state['raw_df'] = read_raw_csv(uploaded, seps=seps)
elif load_example:
    st.session_state['raw_df'] = read_raw_csv(os.path.join("data", "retail_store_inventory.csv"))

if st.session_state['raw_df'] is None:
    st.info("Importez un fichier CSV ou chargez un exemple pour commencer.")
    st.stop()

raw_df = st.session_state['raw_df']

# Quick Stats
c1, c2, c3 = st.columns(3)
with c1: st.metric("Lignes", f"{len(raw_df):,}")
with c2: st.metric("Colonnes", len(raw_df.columns))
with c3: st.metric("Numériques", len(raw_df.select_dtypes(include='number').columns))

st.markdown("---")

# --- Column Configuration ---
st.markdown("### Configuration de l'Analyse")
colSelect1, colSelect2 = st.columns(2)
with colSelect1:
    date_options = ["(Aucune)"] + list(raw_df.columns)
    date_choice = st.selectbox("Colonne Date", date_options, key='date_choice_sel')
with colSelect2:
    val_options = list(raw_df.columns)
    # Try to find a good numeric column by default
    default_val_idx = 0
    numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
         default_val_idx = val_options.index(numeric_cols[0])
    val_choice = st.selectbox("Colonne Cible (Y)", val_options, index=default_val_idx, key='value_choice_sel')

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Aperçu", "Prétraitement", "Anomalies", "Forecast"])

with tab1:
    st.subheader("Aperçu des données brutes")
    st.dataframe(raw_df.head(100), use_container_width=True)

with tab2:
    st.header("Prétraitement")
    p1, p2, p3 = st.columns(3)
    with p1: freq = st.selectbox("Fréquence (Resample)", ['D','W','M'], index=0, key='freq_tab')
    with p2: impute_method = st.selectbox("Imputation", ['interpolate', 'ffill', 'bfill', 'zero'], key='impute_tab')
    with p3: norm_method = st.selectbox("Normalisation", ['Aucune', 'MinMax', 'Standard'], key='norm_tab')

    if st.button("Appliquer le prétraitement", type="primary"):
        with st.spinner("Traitement en cours..."):
            try:
                ts_col = None if date_choice == "(Aucune)" else date_choice
                df_proc = prepare_time_series(raw_df, ts_col=ts_col, value_col=val_choice)
                df_proc = ensure_freq(df_proc, freq=freq)
                df_proc = impute_missing(df_proc, method=impute_method)
                df_proc = cap_outliers_iqr(df_proc, col='y')
                if norm_method != 'Aucune':
                    df_proc = normalize(df_proc, method=norm_method)
                
                st.session_state['df'] = df_proc
                st.success("Données nettoyées et préparées.")
            except Exception as e:
                st.error(f"Erreur de prétraitement: {e}")
                if st.button("Tenter un Auto-Fix"):
                    st.session_state['df'] = auto_fix_time_series(raw_df, ts_col=ts_col, value_col=val_choice)
                    st.rerun()

    if st.session_state['df'] is not None:
        st.subheader("Série après nettoyage")
        df = st.session_state['df']
        st.line_chart(df.set_index('ds')['y'])
        st.dataframe(df.head())

with tab3:
    st.header("Anomalies")
    if st.session_state['df'] is None:
        st.info("Prétraitez d'abord les données.")
    else:
        df = st.session_state['df'].copy()
        threshold = st.slider("Seuil (Z-Score)", 1.0, 5.0, 3.0, 0.5)
        df = zscore_anomalies(df, col='y', threshold=threshold)
        st.write(f"Anomalies détectées : {int(df['anomaly_z'].sum())}")
        plot_anomalies(df, key="anom_plot_tab")

with tab4:
    st.header("Forecast")
    if st.session_state['df'] is None:
        st.info("Prétraitez d'abord les données.")
    else:
        df = st.session_state['df']
        periods = st.number_input("Périodes à prévoir", 1, 365, 30)
        
        if st.button("Lancer le Forecast"):
            if len(df.dropna()) < 2:
                st.error("Pas assez de données valides pour l'entraînement (min 2).")
            else:
                with st.spinner("Entraînement de Prophet..."):
                    model, path = train_prophet(df, save_name='prophet_model.joblib')
                    st.session_state['forecast'] = predict_prophet(model, periods=periods, freq=st.session_state.get('freq_tab', 'D'))
                    st.session_state['model'] = model
                    st.success("Prédiction générée !")

        if st.session_state['forecast'] is not None:
            plot_forecast(df, st.session_state['forecast'], key="forecast_plot_tab")
            if st.checkbox("Voir décomposition"):
                try:
                    comps = decompose_prophet(st.session_state['model'], df)
                    st.line_chart(comps.set_index('ds'))
                except:
                    st.info("Décomposition indisponible.")
