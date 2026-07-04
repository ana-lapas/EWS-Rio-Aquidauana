from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
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
def predict(request: PredictionRequest):
    try:
        # 1. Validação e Cache
        cache_key = f"{request.start_date}_{request.days}"
        if cache_key in cache:
            return {"source": "cache", "data": cache[cache_key]}

        # 2. Pipeline (Simulando carregamento de novos dados da ANA)
        # Em produção: aqui você chamaria a função que busca dados na API HidroWeb
        # e passa pelo DataPipeline.process_new_data()
        
        # 3. Inferência (Instanciando e executando)
        inference = ModelInference(model_path="models/best_model.keras")
        # Simulação de dados de entrada processados
        dummy_input = np.random.rand(1, 30, 9) 
        
        preds = inference.predict_30_days(dummy_input)
        metrics = inference.get_q_metrics(preds)
        
        # 4. Formatação de saída
        result = {
            "previsoes": [float(p) for p in preds.flatten()],
            "metricas": metrics,
            "incerteza_std": float(np.std(preds)),
            "aviso": "Valores representam estimativas baseadas em LSTM com correção de Duan."
        }
        
        cache[cache_key] = result
        return {"source": "model", "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")