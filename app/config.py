import os

def get_pipeline_config():
    """
    Retorna a configuração completa do pipeline.
    Pode ser sobrescrita por variáveis de ambiente se necessário.
    """
    return {
        "drive_path": os.getenv("DATA_RAW_PATH", "data/raw"),
        "output_dir": os.getenv("DATA_PROCESSED_PATH", "data/processed"),
        "output_file": os.getenv("OUTPUT_FILE", "final_dataset_ANA_regressao.csv"),
        "diagnostic_file": os.getenv("DIAGNOSTIC_FILE", "preenchimento_diagnostico.csv"),
        "scaler_file": os.getenv("SCALER_FILE", "data/scaler.pkl"),
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

def is_production():
    """Retorna True se ambiente for 'production'."""
    return os.getenv("ENV", "development").lower() == "production"