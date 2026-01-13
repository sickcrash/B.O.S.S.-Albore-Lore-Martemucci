import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from platform_core import BenchmarkPlatform
from adapters import ResNetAdapter, EfficientNetAdapter
import glob
import subprocess
import sys

st.set_page_config(page_title="B.O.S.S. Platform", layout="wide")

#Blocco utile a nascondere menu di Streamlit
st.markdown("""
    <style>
    /* Nasconde il bottone Deploy */
    .stDeployButton {
        visibility: hidden;
    }

    /* Nasconde il menu con i tre puntini (Hamburger Menu) */
    #MainMenu {
        visibility: hidden;
    }

    /* Opzionale: Nasconde la scritta "Made with Streamlit" in basso */
    footer {
        visibility: hidden;
    }

    /* Opzionale: Nasconde la barra colorata decorativa in cima allo schermo */
    header {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
# -------------------------------------------

st.title("B.O.S.S. Benchmarking Platform")
st.markdown("""
Questa piattaforma analizza e compara le performance di modelli di Computer Vision 
per dispositivi wearable per ipovedenti.
""")

st.divider()

#Sidebar
st.sidebar.header("Configurazione Test")
device_option = st.sidebar.selectbox("Dispositivo Hardware", ["CPU", "GPU (se disponibile)"])
num_images = len(glob.glob("dataset_test/*.jpg"))
st.sidebar.info(f"Dataset caricato: {num_images} immagini")


def run_benchmark_logic():
    progress_bar = st.progress(0)
    status_text = st.empty()

    #Lancio Res
    status_text.text("Avvio Processo Isolato: DeepLabV3 (ResNet50)...")
    try:
        subprocess.run([sys.executable, "run_thesis.py", "--model", "resnet"], check=True)
        status_text.text("ResNet completato. Pulizia RAM...")
    except subprocess.CalledProcessError as e:
        st.error(f"Errore durante il test ResNet: {e}")
        return

    progress_bar.progress(50)
    time.sleep(1)

    #Lancio Efficient
    status_text.text("Avvio Processo Isolato: EfficientNet...")
    try:
        subprocess.run([sys.executable, "run_thesis.py", "--model", "efficientnet"], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Errore durante il test EfficientNet: {e}")
        return

    progress_bar.progress(100)
    status_text.success("Benchmark Completato!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()


#Interfaccia

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Esecuzione Benchmark")
    if st.button(" AVVIA NUOVO TEST", type="primary"):
        with st.spinner('Il sistema sta elaborando i modelli... attendere prego...'):
            run_benchmark_logic()
            st.rerun()  #Ricarico la pagina per mostrare i nuovi dati

with col2:
    st.subheader("2. Stato Sistema")
    if os.path.exists("risultati_resnet.csv") and os.path.exists("risultati_efficientnet.csv"):
        st.success("Dati pronti per l'analisi.")
    else:
        st.warning("Nessun dato trovato. Avvia un test.")

st.divider()

#Dashbord dei risultati
if os.path.exists("risultati_resnet.csv") and os.path.exists("risultati_efficientnet.csv"):

    df_res = pd.read_csv("risultati_resnet.csv")
    df_eff = pd.read_csv("risultati_efficientnet.csv")

    #Calcolo media
    res_lat = df_res["Latency_ms"].mean()
    eff_lat = df_eff["Latency_ms"].mean()
    speedup = res_lat / eff_lat

    #Metriche a confronto
    st.subheader("Confronto Prestazioni")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(label="Latenza ResNet", value=f"{res_lat:.1f} ms", delta="- Lento", delta_color="inverse")
    with kpi2:
        st.metric(label="Latenza EfficientNet", value=f"{eff_lat:.1f} ms", delta=f"+{speedup:.1f}x Veloce")
    with kpi3:
        st.metric(label="Throughput EfficientNet", value=f"{1000 / eff_lat:.1f} FPS")

    #Grafico
    tab1, tab2 = st.tabs(["Grafico Comparativo", "Dati Dettagliati"])

    with tab1:
        #Creiamo un dataFrame per il grafico
        chart_data = pd.DataFrame({
            "Modello": ["ResNet50", "EfficientNet"],
            "Latenza (ms)": [res_lat, eff_lat],
            "RAM (MB)": [df_res["RAM_Total_MB"].mean(), df_eff["RAM_Total_MB"].mean()]
        })

        st.bar_chart(chart_data, x="Modello", y="Latenza (ms)", color=["#FF4B4B"])

    with tab2:
        st.write("Dettaglio predizioni ResNet:")
        st.dataframe(df_res)
        st.write("Dettaglio predizioni EfficientNet:")
        st.dataframe(df_eff)

else:
    st.info("Clicca su 'AVVIA NUOVO TEST' per generare i dati.")