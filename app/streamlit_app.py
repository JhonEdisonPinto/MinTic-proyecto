"""Aplicación Streamlit mínima para visualización rápida.

Comentarios en español: Este archivo sirve como punto de partida para hacer visualizaciones
de los datasets descargados desde datos.gov.co.
"""
import streamlit as st
from dotenv import load_dotenv
import os
from mintic_project.processor import read_csv_url, sample_counts

load_dotenv()

st.set_page_config(page_title="MinTIC - Analítica", layout="wide")

st.title("MinTIC — Explorador de datos abiertos")

DATA_URL = os.getenv("DATA_SOURCE_URL", "")

if not DATA_URL:
    st.warning("No se ha configurado `DATA_SOURCE_URL` en el entorno. Revisar .env")
else:
    with st.spinner("Cargando datos..."):
        df = read_csv_url(DATA_URL, low_memory=False)
    st.success(f"Cargados {len(df):,} registros")
    st.subheader("Conteo por categoría (ejemplo)")
    col = st.selectbox("Seleccionar columna", options=df.columns.tolist())
    n = st.slider("Top N", 1, 50, 10)
    st.dataframe(sample_counts(df, col, n))
