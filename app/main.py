from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from app.etl_pipeline import DataPipeline
from app.model_inference import ModelInference

app = FastAPI(
    title="API de Previsão de Vazões - Rio Aquidauana",
    description="Sistema de suporte à decisão para outorgas e gestão de secas.",
    version="1.0.0"
)

cache = {}

class PredictionRequest(BaseModel):
    start_date: str = Field(..., example="2026-07-01")
    days: int = Field(30, ge=1, le=30)

@app.get("/health")
def health_check():
    return {"status": "online", "model": "LSTM-v8.1-Aquidauana"}

@app.get("/last_update")
def last_update():
    return {"last_data_point": "2026-07-03"}

@app.post("/predict", summary="Prever vazões futuras")
def predict(self, input_sequence, horizon=30):
    if horizonte is not None:
        horizon = horizonte
    # 🔽 Configuração centralizada – igual à usada no Streamlit
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

    pipeline = DataPipeline(config=config)
    df = pipeline.get_latest_data()
    input_seq = pipeline.prepare_input_sequence(df, n_steps=30)
    model = ModelInference(
        model_path="models/modelo_lstm.keras",
        scaler_path=config["scaler_file"]
    )
    preds = model.predict(input_seq, horizon=request.days)
    metrics = model.get_q_metrics(preds)
    return {"previsoes": preds.tolist(), "metricas": metrics}