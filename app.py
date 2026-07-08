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

@st.cache_data(ttl=3600)
def get_metrics():
    pipeline = get_pipeline()
    return pipeline.update_metrics(max_days=365)

# --- Páginas ---
def pagina_monitoramento():
    st.header("📊 Monitoramento de Performance")

    with st.spinner("Calculando métricas de desempenho... (pode levar alguns segundos)"):
        try:
            metrics = get_metrics()
            global_m = metrics['global']
            seasonal = metrics['seasonal']
        except Exception as e:
            st.error(f"Erro ao calcular métricas: {e}")
            st.info("Verifique se o ETL foi executado e o modelo está disponível.")
            return

    # Métricas globais com legendas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NSE (Teste)", f"{global_m['nse']:.4f}")
        st.caption("Eficiência geral do modelo. > 0.75 = muito bom.")
    with col2:
        st.metric("PBIAS", f"{global_m['pbias']}%")
        st.caption("Viés volumétrico. ±10% = excelente.")
    with col3:
        st.metric("RMSE", f"{global_m['rmse']} m³/s")
        st.caption("Erro médio quadrático (absoluto). Quanto menor, melhor.")

    st.subheader("📅 Performance por Período Hidrológico")
    df_seasonal = pd.DataFrame({
        "Período": list(seasonal.keys()),
        "NSE": [v['NSE'] for v in seasonal.values()],
        "NSElog": [v['NSElog'] for v in seasonal.values()]
    })
    st.table(df_seasonal.style.format({"NSE": "{:.4f}", "NSElog": "{:.4f}"}))
    st.caption("NSElog prioriza o ajuste em baixas vazões – essencial para estiagem.")

    st.subheader("🔍 Detalhes das Métricas")
    col_a, col_b = st.columns(2)
    with col_a:
        st.json({
            "Q90 Observado": global_m['q90_obs'],
            "Q90 Simulado": global_m['q90_sim'],
            "Erro Q90": f"{global_m['err_q90']}%",
        })
        st.caption("Q90 = vazão superada em 90% do tempo (referência para outorga).")
    with col_b:
        st.json({
            "Q95 Observado": global_m['q95_obs'],
            "Q95 Simulado": global_m['q95_sim'],
            "Erro Q95": f"{global_m['err_q95']}%",
            "NSE Seca": global_m['nse_seca'],
            "N° observações": global_m['n_obs']
        })
        st.caption("NSE_Seca pode ser negativo devido à baixa variabilidade na estiagem. O importante é o erro Q95 baixo.")

    st.subheader("🔄 Saúde Operacional")
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
            st.subheader("📈 Hidrograma Previsto")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=datas, y=previsoes, mode='lines', name='Vazão Prevista', line=dict(color='#3399ff')))
            fig.update_layout(xaxis_title="Data", yaxis_title="Vazão (m³/s)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("📊 Métricas de Referência")
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

def pagina_sobre():
    st.header("📖 Sobre o Projeto")
    st.markdown("""
    ### 🎯 Objetivo
    Este sistema fornece **previsões diárias de vazões mínimas** para o Rio Aquidauana (MS), utilizando redes neurais LSTM (Long Short-Term Memory) treinadas com 30 anos de dados históricos.

    ### 🧠 Modelo
    - **Arquitetura:** LSTM com 128 neurônios, 2 camadas ocultas, dropout 0.2.
    - **Entrada:** 9 variáveis hidrológicas (vazões e precipitações) + 1 variável cíclica (seno do mês).
    - **Treinamento:** espaço logarítmico, com correção de viés de Duan (smearing factor).
    - **Horizonte de previsão:** até 90 dias (configurável).

    ### 📊 Métricas de Desempenho (valores da dissertação)
    | Métrica | Valor | Interpretação |
    |---------|-------|---------------|
    | **NSE** | 0.8849 | Muito bom (> 0.75) – explica 88% da variabilidade. |
    | **NSElog** | 0.9122 | Excelente em baixas vazões. |
    | **PBIAS** | +1.16% | Viés praticamente nulo (±10% é excelente). |
    | **RMSE** | 25.62 m³/s | Erro absoluto aceitável para a escala da bacia. |
    | **Erro Q90** | 14.26% | Bom – outorga baseada no modelo é confiável. |
    | **Erro Q95** | ~14% | Bom – mesmo para secas severas. |
    | **NSE_Seca** | -3.97 | **Atenção:** Valor negativo é comum em períodos de baixa variabilidade. Não indica erro, apenas que o modelo tem dificuldade em capturar pequenas flutuações na seca. O importante é que o erro Q95 e o PBIAS continuem baixos. |

    ### 🔧 Como usar
    - **Previsão Principal:** escolha a data de início e o horizonte (7 a 90 dias). O gráfico mostra a vazão prevista.
    - **Monitoramento:** veja as métricas de desempenho atualizadas automaticamente com base nos últimos 365 dias de dados.
    - **Atualizar dados da ANA:** força a execução do ETL para incorporar novos dados brutos (arquivos CSV em `data/raw/`).

    ### 📌 Limitações
    - O modelo foi treinado com dados até 2024. Para cenários futuros, o desempenho pode degradar se houver mudanças climáticas ou na bacia.
    - Previsões para horizontes > 30 dias são mais incertas – use com cautela.

    ### 👩‍🔬 Autoria
    **Ana Paula Lapas Leão**
    Orientação: Prof. Dr. Ariel Ortiz Gomes
    Mestrado Profissional em Gestão e Regulação de Recursos Hídricos (PROFÁGUA) – UEMS
    """)

# --- Barra Lateral e Navegação ---
st.sidebar.header("⚙️ Configurações")
menu = st.sidebar.radio("Navegação", ["Previsão Principal", "Monitoramento", "Sobre o Projeto"])
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

# Renderização condicional
if menu == "Previsão Principal":
    pagina_principal(data_inicial, horizon)
elif menu == "Monitoramento":
    pagina_monitoramento()
else:
    pagina_sobre()

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center;'><strong>Produto do Projeto de Mestrado em Rede Nacional Profissionalizante em Gestão e Regulação de Recursos Hídricos (PROFÁGUA)</strong></div>", unsafe_allow_html=True)
footer_col1, footer_col2 = st.columns([1, 1])
footer_col1.markdown("**Desenvolvido por:** Ana Paula Lapas Leão<br>**Orientação:** Prof. Dr. Ariel Ortiz Gomes", unsafe_allow_html=True)
footer_col2.markdown("<div style='text-align: right;'><a href='https://github.com/ana-lapas/EWS-Rio-Aquidauana' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repositório-black?style=for-the-badge&logo=github'></a></div>", unsafe_allow_html=True)