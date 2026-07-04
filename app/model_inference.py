import tensorflow as tf
import numpy as np
import logging

class ModelInference:
    def __init__(self, model_path, smearing_factor=0.994784):
        self.model = tf.keras.models.load_model(model_path)
        self.smearing = smearing_factor
        self.epsilon = 1e-6

    def predict_30_days(self, input_sequence):
        """
        Realiza a inferência e aplica a correção de Duan.
        O modelo original foi treinado em log-space[cite: 1].
        """
        try:
            # Inferência bruto (log-space)
            pred_log_scaled = self.model.predict(input_sequence)

            # O modelo espera retornar ao espaço real:
            # 1. Inverter o Scaler
            # 2. Exponencial
            # 3. Aplicar Smearing Factor

            # Aqui simplificamos assumindo que o modelo já desnormaliza
            # ou que o pipeline de pós-processamento é aplicado após.
            prediction_real = np.exp(pred_log_scaled) - self.epsilon

            # Aplicação do Smearing Factor (correção de viés)[cite: 1]
            corrected_prediction = prediction_real * self.smearing

            return corrected_prediction

        except Exception as e:
            logging.error(f"Erro durante inferência: {e}")
            return None

    def get_q_metrics(self, predictions):
        """Calcula Q90 e Q95 da série prevista."""
        return {
            "Q90": np.percentile(predictions, 10),
            "Q95": np.percentile(predictions, 5)
        }