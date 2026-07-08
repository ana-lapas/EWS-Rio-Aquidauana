import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from app.etl_pipeline import DataPipeline
from app.model_inference import ModelInference

# --- Configuração da Página ---
st.set_page_config(page_title="Gestão de Vazões - Rio Aquidauana", layout="wide")
st.cache_resource.clear()
# --- Funções de Cache (Performance) ---
@st.cache_resource
def get_pipeline():
    config = {
        "drive_path": "data/raw",
        "output_dir": "data/processed",
        "output_file": "final_dataset_ANA_regressao.csv",
        "diagnostic_file": "preenchimento_diagnostico.csv",
        "scaler_file": "data/scaler.pkl",
        "start_date": "1994-02-01",
        "end_date": "2024-01-31",
        "max_missing_pct": 15.0,
        "min_common_pairs": 30,
        "train_frac": 0.70,
        "stations": [
            {"file": "1954002_Chuvas.csv", "name": "Precipitacao_1954002", "type": "Chuva"},
            {"file": "2054019_Chuvas.csv", "name": "Precipitacao_2054019", "type": "Chuva"},
            {"file": "66926000_Vazoes.csv", "name": "Vazao_66926000", "type": "Vazao"},
            {"file": "2054005_Chuvas.csv", "name": "Precipitacao_2054005", "type": "Chuva"},
            {"file": "2054009_Chuvas.csv", "name": "Precipitacao_2054009", "type": "Chuva"},
            {"file": "2055003_Chuvas.csv", "name": "Precipitacao_2055003", "type": "Chuva"},
            {"file": "66941000_Vazoes.csv", "name": "Vazao_66941000", "type": "Vazao"},
            {"file": "2055002_Chuvas.csv", "name": "Precipitacao_2055002", "type": "Chuva"},
            {"file": "66945000_Vazoes.csv", "name": "Vazao_66945000", "type": "Vazao"}
        ],
        "desired_order": [
            "Precipitacao_1954002", "Precipitacao_2054019", "Vazao_66926000",
            "Precipitacao_2054005", "Precipitacao_2054009", "Precipitacao_2055003",
            "Vazao_66941000", "Precipitacao_2055002", "Vazao_66945000"
        ]
    }
    return DataPipeline(config=config)

@st.cache_resource
def get_model():
    config = {
        "scaler_file": "data/scaler.pkl"
    }
    return ModelInference(
        model_path="models/modelo_lstm.keras",
        scaler_path=config["scaler_file"]
    )

@st.cache_data
def load_metrics():
    """Carrega as métricas de desempenho de um arquivo JSON externo."""
    try:
        with open('data/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"global": {"nse": 0.0, "pbias": 0.0, "rmse": 0.0}, "seasonal": {}}

# --- Páginas ---
def pagina_monitoramento():
    st.header("Monitoramento de Performance")
    metrics = load_metrics()

    col1, col2, col3 = st.columns(3)
    col1.metric("NSE (Teste)", f"{metrics['global']['nse']:.4f}")
    col2.metric("PBIAS", f"{metrics['global']['pbias']}%")
    col3.metric("RMSE", f"{metrics['global']['rmse']}")

    st.subheader("Performance por Período Hidrológico")
    st.table(pd.DataFrame(metrics.get('seasonal', {})).T)

    st.subheader("Saúde Operacional")
    media_treino, std_treino = 25.0, 5.0
    precipitacao_atual = 150
    if precipitacao_atual > (media_treino + 2 * std_treino):
        st.warning("⚠️ Alerta: Entrada de precipitação fora da distribuição de treinamento (Drift).")
    else:
        st.success("✅ Modelo operando dentro da distribuição esperada.")


def pagina_principal(data_inicial, horizon):
    try:
        pipeline = get_pipeline()
        model = get_model()
        with st.spinner('Carregando dados e realizando previsão...'):
            df_historico = pipeline.get_latest_data()
            input_seq = pipeline.prepare_input_sequence(df_historico, n_steps=30)
            previsoes = model.predict(input_seq, horizon=horizon)

        datas = pd.date_range(data_inicial, periods=horizon)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Hidrograma Previsto")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=datas, y=previsoes, mode='lines', name='Vazão Prevista', line=dict(color='#3399ff')))
            fig.update_layout(xaxis_title="Data", yaxis_title="Vazão (m³/s)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Métricas de Referência")
            q90 = np.percentile(previsoes, 10)
            q95 = np.percentile(previsoes, 5)
            st.metric("Q90 Estimado", f"{q90:.2f} m³/s")
            st.metric("Q95 Estimado", f"{q95:.2f} m³/s")
            if q95 < 10:
                st.error("⚠️ ALERTA: Vazão abaixo do limiar crítico (Q95)")
            else:
                st.info("✅ Nível operacional estável.")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("Verifique se os arquivos brutos estão em 'data/raw/' e execute o ETL manualmente.")

# --- Barra Lateral e Navegação ---
st.sidebar.header("Configurações")
menu = st.sidebar.radio("Navegação", ["Previsão Principal", "Monitoramento"])
data_inicial = st.sidebar.date_input("Data de início", datetime.today())
horizon = st.sidebar.slider("horizon de previsão (dias)", 7, 90, 30)

if st.sidebar.button("Atualizar dados da ANA"):
    with st.spinner('Executando ETL...'):
        pipeline = get_pipeline()
        pipeline.run_etl()
        st.success("Dados atualizados com sucesso!")

# --- Renderização Condicional ---
if menu == "Previsão Principal":
    pagina_principal(data_inicial, horizon)
else:
    pagina_monitoramento()

# --- Rodapé Institucional ---
st.markdown("---")
st.markdown("<div style='text-align: center;'><strong>Produto do Projeto de Mestrado em Rede Nacional Profissionalizante em Gestão e Regulação de Recursos Hídricos (PROFAGUA)</strong></div>", unsafe_allow_html=True)

footer_col1, footer_col2 = st.columns([1, 1])
footer_col1.markdown("**Desenvolvido por:** Ana Paula Lapas Leão<br>**Orientação:** Prof. Dr. Ariel Ortiz Gomes", unsafe_allow_html=True)
footer_col2.markdown("<div style='text-align: right;'><a href='https://github.com/ana-lapas/EWS-Rio-Aquidauana' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repositório-black?style=for-the-badge&logo=github'></a></div>", unsafe_allow_html=True)