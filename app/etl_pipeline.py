import pandas as pd
import numpy as np
import logging

class DataPipeline:
    def __init__(self, scaler, config):
        self.scaler = scaler
        self.config = config
        self.epsilon = 1e-6

    def process_new_data(self, df_raw):
        """
        Limpeza e preparação para inferência.
        df_raw deve conter as colunas na ordem original do treinamento.
        """
        try:
            # 1. Log Transform
            df_log = np.log(np.maximum(df_raw, self.epsilon))

            # 2. Normalização (usando o scaler do treino)
            scaled_data = self.scaler.transform(df_log)

            return scaled_data
        except Exception as e:
            logging.error(f"Erro no processamento de novos dados: {e}")
            raise

    def create_inference_sequence(self, processed_data, n_steps=30):
        """Prepara a última janela de 30 dias para o LSTM."""
        if len(processed_data) < n_steps:
            raise ValueError(f"Dados insuficientes. Necessário {n_steps} dias.")
        sequence = processed_data[-n_steps:]
        return np.expand_dims(sequence, axis=0)