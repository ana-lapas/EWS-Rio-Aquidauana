import tensorflow as tf
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path, scaler_path="data/scaler.pkl", smearing_factor=0.994784):
        self.model = tf.keras.models.load_model(model_path)
        self.smearing = smearing_factor
        self.epsilon = 1e-6

        # Carregar scaler (treinado apenas com as 9 features hidrológicas)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.n_features_scaler = self.scaler.n_features_in_  # = 9

        # Índice da variável alvo nas 9 features (Vazao_66945000 é a última)
        self.target_idx = 8  # 0-based (9ª coluna)

        # Quantas features cíclicas o modelo espera?
        # model.input_shape[-1] é o total de features (ex: 10 ou 11)
        self.n_cyclic = self.model.input_shape[-1] - self.n_features_scaler

    def predict(self, input_sequence, horizon=30):
        """
        input_sequence: numpy array shape (1, n_steps, n_features_total)
                        já normalizado (log1p + scaler) + cíclicas.
        """
        preds = []
        current_seq = input_sequence.copy()

        for _ in range(horizon):
            # 1. Prever próximo passo (escala normalizada)
            pred_log_scaled = self.model.predict(current_seq, verbose=0).flatten()

            # 2. Desnormalizar usando o scaler (apenas 9 features)
            dummy = np.zeros((1, self.n_features_scaler))
            dummy[0, self.target_idx] = pred_log_scaled[0]
            pred_log = self.scaler.inverse_transform(dummy)[0, self.target_idx]

            # 3. Converter para vazão real (m³/s)
            pred_real = np.exp(pred_log) - self.epsilon
            pred_real *= self.smearing
            preds.append(pred_real)

            # 4. Atualizar sequência para o próximo passo
            new_step = current_seq[0, -1, :].copy()
            new_step[self.target_idx] = pred_log_scaled[0]  # atualiza apenas a target (dentro das 9)
            # As cíclicas permanecem iguais
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_step

        return np.array(preds)

    def get_q_metrics(self, predictions):
        return {
            "Q90": np.percentile(predictions, 10),
            "Q95": np.percentile(predictions, 5)
        }