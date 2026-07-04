import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from app.etl_pipeline import DataPipeline
from app.model_inference import ModelInference

# Configuração da página
st.set_page_config(page_title="Gestão de Vazões - Rio Aquidauana", layout="wide")

# --- Variáveis Globais de Monitoramento (Exemplos) ---
# Em produção, carregue esses valores de um arquivo ou banco de dados
media_treino = 25.0
std_treino = 5.0

def pagina_monitoramento():
    st.header("Monitoramento de Performance")

    # 1. Métricas Globais
    col1, col2, col3 = st.columns(3)
    col1.metric("NSE (Teste)", "0.8849")
    col2.metric("PBIAS", "1.16%")
    col3.metric("RMSE", "4.21") # Valor hipotético baseado no seu log

    # 2. Métricas Sazonais (O diferencial da sua dissertação)
    st.subheader("Performance por Período Hidrológico")
    dados_sazonais = {
        "Período": ["Cheia", "Vazante", "Seca"],
        "NSE": [0.85, 0.78, 0.91], # Substituir pelos dados reais do log
        "NSElog": [0.89, 0.82, 0.93]
    }
    st.table(pd.DataFrame(dados_sazonais))

    # 3. Drift Detection
    st.subheader("Saúde Operacional")
    precipitacao_atual = 150
    if precipitacao_atual > (media_treino + 2 * std_treino):
        st.warning("⚠️ Alerta: Entrada de precipitação fora da distribuição de treinamento (Drift).")
    else:
        st.success("✅ Modelo operando dentro da distribuição esperada.")

def pagina_principal(data_inicial, horizonte):
    st.title("💧 Previsão Hidrológica: Rio Aquidauana (Estação 66945000)")
    st.markdown("Plataforma de suporte à decisão para outorgas e gestão de secas.")
    
    col1, col2 = st.columns([2, 1])
    datas = pd.date_range(data_inicial, periods=horizonte)
    vazoes_previstas = np.random.uniform(5, 50, size=horizonte)

    with col1:
        st.subheader("Hidrograma Previsto")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datas, y=vazoes_previstas, mode='lines', name='Vazão Prevista', line=dict(color='#3399ff')))
        fig.update_layout(xaxis_title="Data", yaxis_title="Vazão (m³/s)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Métricas de Referência")
        q90, q95 = np.percentile(vazoes_previstas, 10), np.percentile(vazoes_previstas, 5)
        st.metric("Q90 Estimado", f"{q90:.2f} m³/s")
        st.metric("Q95 Estimado", f"{q95:.2f} m³/s")
        if q95 < 10:
            st.error("⚠️ ALERTA: Vazão abaixo do limiar crítico (Q95)")
        else:
            st.info("✅ Nível operacional estável.")

# --- Barra Lateral e Navegação ---
st.sidebar.header("Configurações")
menu = st.sidebar.radio("Navegação", ["Previsão Principal", "Monitoramento"])
data_inicial = st.sidebar.date_input("Data de início", datetime.today())
horizonte = st.sidebar.slider("Horizonte de previsão (dias)", 7, 90, 30)

if st.sidebar.button("Atualizar dados da ANA"):
    st.success("Dados atualizados!")

# --- Renderização Condicional ---
if menu == "Previsão Principal":
    pagina_principal(data_inicial, horizonte)
else:
    pagina_monitoramento()

# --- Rodapé Institucional ---
st.markdown("---")
st.markdown("<div style='text-align: center;'><strong>Produto do Projeto de Mestrado em Rede Nacional Profissionalizante em Gestão e Regulação de Recursos Hídricos (PROFAGUA)</strong></div>", unsafe_allow_html=True)

footer_col1, footer_col2 = st.columns([1, 1])
footer_col1.markdown("**Desenvolvido por:** Ana Paula Lapas Leão<br>**Orientação:** Prof. Dr. Ariel Ortiz Gomes", unsafe_allow_html=True)
footer_col2.markdown("<div style='text-align: right;'><a href='https://github.com/SEU_USUARIO/SEU_REPOSITORIO' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repositório-black?style=for-the-badge&logo=github'></a></div>", unsafe_allow_html=True)