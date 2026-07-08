from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from app.etl_pipeline import DataPipeline
from app.model_inference import ModelInference
from app.config import get_pipeline_config

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

@app.post("/predict")
def predict(request: PredictionRequest):
    config = get_pipeline_config()  # ← IMPORTANTE: definir config
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

@app.post("/update_metrics")
def update_metrics():
    pipeline = DataPipeline(config=get_pipeline_config())
    result = pipeline.update_metrics()
    return {"status": "success", "metrics": result}