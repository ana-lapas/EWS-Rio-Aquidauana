import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def calculate_metrics(obs, sim):
    """
    Calcula métricas hidrológicas padrão.
    
    Args:
        obs: array observado
        sim: array simulado
    
    Returns:
        dict: dicionário com métricas
    """
    if len(obs) == 0 or np.std(obs) == 0:
        return {k: np.nan for k in ['RMSE','NSE','NSElog','PBIAS',
                'Q90_Obs','Q95_Obs','Q90_Sim','Q95_Sim','ErrQ90','ErrQ95','NSE_Seca']}
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(obs, sim))
    
    # NSE
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    
    # PBIAS (sinal correto: positivo = superestimativa, negativo = subestimativa)
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)
    
    # Truncar valores negativos para métricas logarítmicas e percentis
    sim_trunc = np.maximum(sim, 0)
    epsilon = 1e-6
    obs_log = np.log(np.maximum(obs, epsilon))
    sim_log = np.log(np.maximum(sim_trunc, epsilon))
    nse_log = (1 - np.sum((obs_log - sim_log)**2) / np.sum((obs_log - np.mean(obs_log))**2)) if np.std(obs_log) > 0 else np.nan
    
    # Percentis
    q90_obs, q95_obs = np.percentile(obs, 10), np.percentile(obs, 5)
    q90_sim, q95_sim = np.percentile(sim_trunc, 10), np.percentile(sim_trunc, 5)
    err_q90 = 100 * (q90_sim - q90_obs) / q90_obs if q90_obs > 0 else np.nan
    err_q95 = 100 * (q95_sim - q95_obs) / q95_obs if q95_obs > 0 else np.nan
    
    # NSE durante períodos de seca (vazões <= Q90)
    idx_seca = obs <= q90_obs
    if np.sum(idx_seca) > 1 and np.std(obs[idx_seca]) > 0:
        nse_seca = 1 - np.sum((obs[idx_seca] - sim_trunc[idx_seca])**2) / np.sum((obs[idx_seca] - np.mean(obs[idx_seca]))**2)
    else:
        nse_seca = np.nan
    
    return {
        'RMSE': rmse,
        'NSE': nse,
        'NSElog': nse_log,
        'PBIAS': pbias,
        'Q90_Obs': q90_obs,
        'Q95_Obs': q95_obs,
        'Q90_Sim': q90_sim,
        'Q95_Sim': q95_sim,
        'ErrQ90': err_q90,
        'ErrQ95': err_q95,
        'NSE_Seca': nse_seca
    }

def calculate_seasonal_metrics(obs, sim, dates):
    """
    Calcula métricas separadas por período hidrológico.
    
    Args:
        obs: array observado
        sim: array simulado
        dates: array de datas (datetime ou string)
    
    Returns:
        dict: {'Cheia': {'NSE': ..., 'NSElog': ..., 'n_dias': ...}, ...}
    """
    dt = pd.to_datetime(dates)
    seasons = {
        'Cheia': [12, 1, 2, 3],
        'Vazante': [4, 5, 6],
        'Seca': [7, 8, 9, 10, 11]
    }
    results = {}
    epsilon = 1e-6
    for season_name, months in seasons.items():
        mask = np.isin(dt.month, months)
        if mask.sum() > 1 and np.std(obs[mask]) > 0:
            obs_s = obs[mask]
            sim_s = np.maximum(sim[mask], 0)
            nse_s = 1 - np.sum((obs_s - sim_s)**2) / np.sum((obs_s - np.mean(obs_s))**2)
            obs_log = np.log(np.maximum(obs_s, epsilon))
            sim_log = np.log(np.maximum(sim_s, epsilon))
            nselog_s = (1 - np.sum((obs_log - sim_log)**2) / np.sum((obs_log - np.mean(obs_log))**2)) if np.std(obs_log) > 0 else np.nan
            results[season_name] = {'NSE': round(nse_s, 4), 'NSElog': round(nselog_s, 4), 'n_dias': int(mask.sum())}
        else:
            results[season_name] = {'NSE': np.nan, 'NSElog': np.nan, 'n_dias': int(mask.sum())}
    return results