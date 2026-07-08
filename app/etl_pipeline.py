import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ETLProcessor:
    """Adaptação do script ETL v3.3 para uso local (sem Google Drive)"""
    def __init__(self, config):
        self.config = config
        self.master_range = pd.date_range(start=config["start_date"],
                                          end=config["end_date"], freq='D')
        self.final_df = None
        self.fill_report = []
        self.train_cutoff = int(len(self.master_range) * config["train_frac"])

    def process_station(self, info):
        """Lê e transforma arquivo CSV da ANA para série temporal."""
        file_path = os.path.join(self.config["drive_path"], info["file"])
        if not os.path.exists(file_path):
            logger.warning(f"Arquivo não encontrado: {info['file']}")
            return None

        skip = 14 if info["type"] == "Chuva" else 15
        df = pd.read_csv(file_path, sep=';', skiprows=skip, encoding='latin1', low_memory=False)

        prefix = info["type"]
        val_cols = [f"{prefix}{str(i).zfill(2)}" for i in range(1, 32)]

        df_long = df.melt(id_vars=['Data'], value_vars=val_cols,
                          var_name='Day_Str', value_name='val')
        df_long['val'] = pd.to_numeric(df_long['val'].astype(str).str.replace(',', '.'),
                                       errors='coerce')
        df_long['date'] = pd.to_datetime(df_long['Data'], format='%d/%m/%Y', errors='coerce')
        df_long['Day'] = df_long['Day_Str'].str.extract(r'(\d+)').astype(int)
        df_long['date'] = df_long['date'] + pd.to_timedelta(df_long['Day'] - 1, unit='D')

        df_clean = df_long.dropna(subset=['date']).drop_duplicates('date')
        merged = pd.DataFrame({'date': self.master_range}).merge(
            df_clean[['date', 'val']], on='date', how='left'
        )
        return merged.set_index('date')['val']

    def _station_ok(self, info):
        """Verifica se a estação tem ≤ max_missing_pct% de falhas."""
        series = self.process_station(info)
        if series is None:
            return False, None
        missing_pct = series.isna().mean() * 100
        return missing_pct <= self.config["max_missing_pct"], series

    def fill_with_regression(self):
        """Preenchimento por REGRESSÃO LINEAR usando apenas dados de treino."""
        df = self.final_df.copy()
        train_mask = df.index < df.index[self.train_cutoff]

        for target in df.columns:
            missing_mask = df[target].isna()
            missing_count = missing_mask.sum()
            if missing_count == 0:
                self.fill_report.append([target, 0, 0, 0, "completo", ""])
                continue

            tipo = 'Precipitacao' if target.startswith('Precipitacao') else 'Vazao' if target.startswith('Vazao') else None
            pred_cols = [c for c in df.columns if c.startswith(tipo) and c != target] if tipo else []

            if not pred_cols:
                mean_val = df.loc[train_mask, target].mean()
                df[target] = df[target].fillna(mean_val)
                self.fill_report.append([target, missing_count, 0, missing_count, "media_propria", ""])
                logger.info(f"{target}: preenchida com média do treino (sem preditoras)")
                continue

            regr_coeffs = {}
            for pred in pred_cols:
                common = df[target].notna() & df[pred].notna() & train_mask
                if common.sum() < self.config["min_common_pairs"]:
                    continue
                x = df.loc[common, pred].values
                y = df.loc[common, target].values
                a, b = np.polyfit(x, y, 1)
                corr = np.corrcoef(x, y)[0, 1]
                regr_coeffs[pred] = (a, b, corr)

            ordered_preds = sorted(regr_coeffs.keys(), key=lambda p: regr_coeffs[p][2], reverse=True)

            filled = df[target].copy()
            filled_by_regr = 0
            filled_by_mean = 0

            for idx in df.index[missing_mask]:
                value_filled = False
                for pred in ordered_preds:
                    pred_val = df.loc[idx, pred]
                    if pd.notna(pred_val):
                        a, b, _ = regr_coeffs[pred]
                        estimated = a * pred_val + b
                        filled.loc[idx] = max(0, estimated)
                        filled_by_regr += 1
                        value_filled = True
                        break
                if not value_filled:
                    mean_target = df.loc[train_mask, target].mean()
                    filled.loc[idx] = max(0, mean_target)
                    filled_by_mean += 1

            df[target] = filled
            metodo = "regressao" if filled_by_regr > 0 else "media_propria"
            self.fill_report.append([target, missing_count, filled_by_regr, filled_by_mean,
                                     metodo, ", ".join(ordered_preds[:3])])
            logger.info(f"{target}: {missing_count} falhas → {filled_by_regr} por regressão, "
                       f"{filled_by_mean} pela média (treino).")

        self.final_df = df

    def run(self):
        """Executa o pipeline ETL completo."""
        logger.info("=== INÍCIO DO PROCESSAMENTO ETL v3.3 - REGRESSÃO LINEAR (LOCAL) ===")
        os.makedirs(self.config["drive_path"], exist_ok=True)

        self.final_df = pd.DataFrame(index=self.master_range)
        included = []

        for info in self.config["stations"]:
            ok, series = self._station_ok(info)
            if not ok:
                logger.warning(f"Estação {info['name']} excluída (> {self.config['max_missing_pct']}% falhas)")
                continue
            self.final_df[info["name"]] = series
            included.append(info["name"])

        if not included:
            logger.error("Nenhuma estação aprovada no filtro!")
            raise RuntimeError("Nenhuma estação disponível para processamento.")

        # Ordenação fisiográfica
        final_order = [name for name in self.config["desired_order"] if name in included]
        self.final_df = self.final_df[final_order]

        self.fill_with_regression()

        self.final_df.index.name = "Datas"
        output_dir = self.config.get("output_dir", "data/processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.config["output_file"])
        print(f"📁 Salvando em: {output_path}")
        self.final_df.to_csv(output_path, index=True)

        # Diagnóstico
        diag_path = os.path.join(output_dir, self.config["diagnostic_file"])
        pd.DataFrame(self.fill_report,
                    columns=["Estacao", "Falhas_antes", "Preenchidas_regressao",
                            "Preenchidas_media", "Metodo", "Preditora"]
                    ).to_csv(diag_path, index=False)
        logger.info(f"Diagnóstico salvo em: {diag_path}")


class DataPipeline:
    def __init__(self, scaler=None, config=None):
        self.scaler = scaler
        self.epsilon = 1e-6
        self.logger = logger

        self.config = config or {
            "drive_path": "data/raw",
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

        self.master_range = pd.date_range(start=self.config["start_date"], end=self.config["end_date"], freq='D')
        output_dir = self.config.get("output_dir", "data/processed")
        self.final_file = os.path.join(output_dir, self.config["output_file"])
        self.scaler_file = self.config.get("scaler_file", "data/processed/scaler.pkl")
        self._ensure_etl()

    def _ensure_etl(self):
        """Executa o ETL e treina o scaler se o arquivo final não existir."""
        self.logger.info(f"Verificando arquivo: {self.final_file}")
        if not os.path.exists(self.final_file):
            self.logger.warning("Arquivo final não encontrado. Executando ETL...")
            try:
                etl = ETLProcessor(self.config)
                etl.run()
                self._train_and_save_scaler()
            except Exception as e:
                self.logger.error(f"ETL falhou: {e}")
                # Opcional: gerar dados sintéticos para demonstração
                # self._generate_synthetic_data()
                raise  # relança para que o usuário veja o erro
        else:
            self.logger.info("Arquivo final encontrado.")

    def _train_and_save_scaler(self):
        """Treina o scaler com os dados de treino (log1p) e salva."""
        df = pd.read_csv(self.final_file, index_col='Datas', parse_dates=True)
        train_size = int(len(df) * self.config["train_frac"])
        train_df = df.iloc[:train_size]
        self.scaler = StandardScaler()
        self.scaler.fit(np.log1p(train_df))
        with open(self.scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)
        self.logger.info("Scaler treinado e salvo em %s", self.scaler_file)

    def get_latest_data(self):
        """Retorna o DataFrame completo; se não existir, tenta ETL novamente."""
        if not os.path.exists(self.final_file):
            self.logger.warning("Arquivo final não encontrado em get_latest_data. Executando ETL...")
            self._ensure_etl()  # tenta novamente
        return pd.read_csv(self.final_file, index_col='Datas', parse_dates=True)

    def prepare_input_sequence(self, df, n_steps=30):
        if self.scaler is None:
            if os.path.exists(self.scaler_file):
                with open(self.scaler_file, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                self._train_and_save_scaler()

        last_df = df.iloc[-n_steps:].copy()
        dates = last_df.index

        # Normalizar as 9 features hidrológicas
        scaled = self.scaler.transform(np.log1p(last_df))  # shape (n_steps, 9)

        # Adicionar feature cíclica (seno do mês) → total 10
        month_sin = np.sin(2 * np.pi * dates.month / 12).values.reshape(-1, 1)
        full_input = np.hstack([scaled, month_sin])  # (n_steps, 10)

        return np.expand_dims(full_input, axis=0)  # (1, n_steps, 10)

    def run_etl(self):
        """Força a execução do ETL e retreina o scaler."""
        etl = ETLProcessor(self.config)
        etl.run()
        self._train_and_save_scaler()