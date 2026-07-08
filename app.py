import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from app.etl_pipeline import DataPipeline
from app.model_inference import ModelInference
from app.config import get_pipeline_config, is_production

st.set_page_config(page_title="Gestão de Vazões - Rio Aquidauana", layout="wide")

if not is_production():
    st.cache_resource.clear()

@st.cache_resource
def get_pipeline():
    return DataPipeline(config=get_pipeline_config())

@st.cache_resource
def get_model():
    config = get_pipeline_config()
    return ModelInference(
        model_path="models/modelo_lstm.keras",
        scaler_path=config["scaler_file"]
    )

@st.cache_data(ttl=3600)  # cache por 1 hora
def get_metrics():
    """Calcula métricas dinamicamente usando o pipeline e o modelo."""
    pipeline = get_pipeline()
    return pipeline.update_metrics(max_days=365)

# --- Páginas ---
def pagina_monitoramento():
    st.header("Monitoramento de Performance")

    # Carregar métricas automaticamente (com spinner)
    with st.spinner("Calculando métricas de desempenho... (pode levar alguns segundos)"):
        try:
            metrics = get_metrics()
            global_m = metrics['global']
            seasonal = metrics['seasonal']
        except Exception as e:
            st.error(f"Erro ao calcular métricas: {e}")
            st.info("Verifique se o ETL foi executado e o modelo está disponível.")
            # Mostrar métricas de fallback (opcional)
            return

    col1, col2, col3 = st.columns(3)
    col1.metric("NSE (Teste)", f"{global_m['nse']:.4f}")
    col2.metric("PBIAS", f"{global_m['pbias']}%")
    col3.metric("RMSE", f"{global_m['rmse']} m³/s")

    st.subheader("Performance por Período Hidrológico")
    df_seasonal = pd.DataFrame({
        "Período": list(seasonal.keys()),
        "NSE": [v['NSE'] for v in seasonal.values()],
        "NSElog": [v['NSElog'] for v in seasonal.values()]
    })
    st.table(df_seasonal)

    st.subheader("Detalhes das Métricas")
    st.json({
        "Q90 Observado": global_m['q90_obs'],
        "Q90 Simulado": global_m['q90_sim'],
        "Erro Q90": f"{global_m['err_q90']}%",
        "Q95 Observado": global_m['q95_obs'],
        "Q95 Simulado": global_m['q95_sim'],
        "Erro Q95": f"{global_m['err_q95']}%",
        "NSE Seca": global_m['nse_seca'],
        "Número de observações": global_m['n_obs']
    })

    st.subheader("Saúde Operacional")
    if st.button("Recalcular métricas agora"):
        st.cache_data.clear()
        st.rerun()

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
horizon = st.sidebar.slider("Horizonte de previsão (dias)", 7, 90, 30)

if st.sidebar.button("Atualizar dados da ANA"):
    with st.spinner('Executando ETL...'):
        try:
            pipeline = get_pipeline()
            pipeline.run_etl()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Dados atualizados com sucesso! Recarregue a página.")
        except Exception as e:
            st.error(f"Erro ao executar ETL: {e}")

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