# pylint: disable=import-error
# pylint: disable=wrong-import-position
"""
title: real_ETT_exp
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a real experiment with a ETT dataset.
The experiment consists of generating a ETT dataset with a polinomial function and
adding Gaussian noise to it. Then, a neural network model is trained to predict the target
variable. Finally, the gradients are used to reduce the noise in the data.
"""

# Libraries & Global variables #
# ------------------------------------------------------------------------------------------------ #
# Public libraries
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global variables
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'time_series', 'real', 'ETT')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', 'time_series', 'real', 'ETT')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'time_series', 'real', 'ETT')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Show info on the terminal about how the execution is going.
VERBOSE = True
# Even if there is a checkpoint, the model is retrained.
FORCE_TRAINING_PRE_XAI = True
FORCE_TRAINING_NN = True
FORCE_TRAINING_POST_XAI = True
# Name of this experiment that will appear in the result files.
SUBFIX_NAME = 'kalman'
IS_TS = True

# Local libraries
from dataset import SlidingWindowDataset
from models import Trainer, XAI_benchmark, LSTMModel
from dlnr import DLNoiseReduction
from utils import symmetric_mean_absolute_percentage_error

# Make sure that the GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"


# Functions definition #
# ------------------------------------------------------------------------------------------------ #
def dictionary_arrays_to_list(array_d:dict):
    """
    Run through a array_d dictionary converting its arrays to lists.

    Args:
        array_d (dict): dictionary which arrays we want to convert to lists.

    Returns:
        dict: dictionary with lists intead of arrays.
    """
    if isinstance(array_d, dict):
        return {k: dictionary_arrays_to_list(v) for k, v in array_d.items()}
    elif isinstance(array_d, np.ndarray):
        return array_d.tolist()
    elif isinstance(array_d, pd.Series):
        return array_d.to_list()
    else:
        return array_d


# Main #
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
    ## Load the data ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = {}
    metrics_dict = {}

    df_data = pd.read_parquet(
        os.path.join(
            DATA_PATH,
            'clean_h1.parquet'
        )
    )

    df_data.rename(columns={"OT": "y"}, inplace=True)

    # scale the data
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

    # assert there is no more categorical variables
    columnas_categoricas = df_data.select_dtypes(include=['object', 'category']).columns
    assert not list(columnas_categoricas)

    # Desplazar la última columna hacia arriba
    df_data['y_shifted'] = df_data['y'].shift(-1)
    # Eliminar la última fila porque tendrá un NaN en la última columna
    df_data = df_data.dropna().reset_index(drop=True)
    y_shifted = df_data['y_shifted'].copy()
    df_data = df_data.drop(columns=['y_shifted'])

    # divide the data into train/test datasets
    input_vars = df_data.columns
    X_train, X_test, y_train, y_test = train_test_split(
        df_data[input_vars].values, y_shifted, test_size=0.2, shuffle=False
    )

    ## Perform XAI benchmark over original data ##
    ## ------------------------------------------------------------------------------------------ ##
    model_params = {
        'ridge': {"alpha": 1.0},
        'pls': {"n_components": 1},
        'tree': {"max_depth": 5},
        'svm': {"kernel": 'poly', "degree": 2},
        'knn': {
            "n_neighbors": 5,
            "weights": 'uniform',
            "algorithm": 'auto',
            "leaf_size": 30,
            "p": 2,
            "n_jobs": None
        },
        'auto_arima': None,
        # 'auto_arima': {
        #     'y': y_train,
        #     'seasonal': True,
        #     'm': 30,
        #     'start_p': 5,
        #     'max_p': 10,
        #     'start_q': 0,
        #     'max_q': 0,
        #     'start_P': 0,
        #     'start_Q': 1,
        #     'max_P': 0,
        #     'max_Q': 1,
        #     'stepwise': True,
        #     'trace': True,
        #     'parallel': True
        # },
        # 'arima': None,
        'arima': {
            'order': (1, 1, 0),
            'seasonal_order': (1, 0, 1, 24)
        }
    }
    xai_benchmark_orig = XAI_benchmark(
        is_ts = IS_TS,
        model_params = model_params,
        verbose = VERBOSE
    )

    if FORCE_TRAINING_PRE_XAI:
        xai_benchmark_orig.fit(X_train, y_train)
    else:
        xai_benchmark_orig.load(os.path.join(CHECKPOINT_PATH, 'orig'))

    orig_pred, orig_metrics = xai_benchmark_orig.predict(
        X_test,
        y_test,
        n_periods=len(y_test),
        get_metrics=True,
        rolling_forcast=False
    )
    xai_benchmark_orig.save(
        path = os.path.join(CHECKPOINT_PATH, 'orig'),
        subfix = 'orig'
    )

    predictions_dict['orig'] = orig_pred
    metrics_dict['orig'] = orig_metrics


    ## Calculate orignal histograms and correlation matrix ##
    ## ------------------------------------------------------------------------------------------ ##
    orig_corr = df_data.corr()
    histogram_orig = {}
    histo_bins_orig = {}
    for col in df_data.columns:
        hist, bin_edges = np.histogram(df_data[col], bins='auto', density=True)
        histogram_orig[col] = hist + 1e-10
        histo_bins_orig[col] = len(bin_edges) - 1


    ## Denoise the data using Kalman Filter ##
    ## ------------------------------------------------------------------------------------------ ##
    filtered_signal = []
    # Inicialización del Filtro de Kalman
    F = 1  # Matriz de transición (1D, sin dinámica compleja)
    H = 1  # Matriz de observación
    Q = 0.01  # Varianza del ruido del proceso
    R = 0.01  # Varianza del ruido de medición
    x = 0  # Estado inicial
    P = 1  # Varianza inicial

    # Filtro de Kalman
    for z in df_data.values:
        # Predicción
        x_pred = F * x
        P_pred = F * P * F + Q

        # Actualización
        K = P_pred * H / (H * P_pred * H + R)  # Ganancia de Kalman
        x = x_pred + K * (z - H * x_pred)
        P = (1 - K * H) * P_pred

        # Guardar el estado filtrado
        filtered_signal.append(x)

    df_denoised = pd.DataFrame(filtered_signal, columns = df_data.columns)
    df_denoised = df_denoised.copy()

    # Calc the metrics
    gt_values = df_data.values
    predicted_values = df_denoised.values
    mae = mean_absolute_error(gt_values, predicted_values)
    smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    denoised_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'R2': r_squared
    }

    predictions_dict['orig'][SUBFIX_NAME] = predicted_values
    metrics_dict['orig'][SUBFIX_NAME] = denoised_metrics

    denoised_corr = df_denoised.corr()
    # Desplazar la última columna hacia arriba
    df_denoised['y_shifted'] = df_denoised['y'].shift(-1)
    # Eliminar la última fila porque tendrá un NaN en la última columna
    df_denoised = df_denoised.dropna().reset_index(drop=True)
    y_denoised_shifted = df_denoised['y_shifted'].copy()
    df_denoised = df_denoised.drop(columns=['y_shifted'])

    # divide the data into train/test datasets
    input_vars = df_data.columns
    X_train_denoised, X_test_denoised, y_train_denoised, y_test_denoised = train_test_split(
        df_denoised[input_vars].values, y_denoised_shifted, test_size=0.2, shuffle=False
    )

    ## Perform XAI benchmark over Denoised data ##
    ## ------------------------------------------------------------------------------------------ ##
    # order = xai_benchmark_orig.auto_arima.order # (p, d, q)
    # seasonal_order = xai_benchmark_orig.auto_arima.seasonal_order # (P, D, Q, m)
    model_params = {
        'ridge': {"alpha": 1.0},
        'pls': {"n_components": 1},
        'tree': {"max_depth": 5},
        'svm': {"kernel": 'poly', "degree": 2},
        'knn': {
            "n_neighbors": 5,
            "weights": 'uniform',
            "algorithm": 'auto',
            "leaf_size": 30,
            "p": 2,
            "n_jobs": None
        },
        'auto_arima': None,
        # 'arima': None,
        # 'arima': {
        #     'order': order,
        #     'seasonal_order': seasonal_order
        # },
        'arima': {
            'order': (1, 1, 0),
            'seasonal_order': (1, 0, 1, 24)
        }
    }

    xai_benchmark_denoised = XAI_benchmark(
        is_ts = IS_TS,
        model_params = model_params,
        verbose = VERBOSE
    )

    if FORCE_TRAINING_POST_XAI:
        xai_benchmark_denoised.fit(X_train_denoised, y_train_denoised)
    else:
        xai_benchmark_denoised.load(os.path.join(CHECKPOINT_PATH, 'denoised'))

    xai_benchmark_denoised.save(
        path = os.path.join(CHECKPOINT_PATH, 'denoised'),
        subfix = f'{SUBFIX_NAME}_denoised'
    )

    # Get the predictions and metrics. Denoised models over denoised data.
    pred_over_denoised, metric_over_denoised = xai_benchmark_denoised.predict(
        X_test_denoised,
        y_test_denoised,
        n_periods=len(y_test_denoised),
        get_metrics=True,
        rolling_forcast=False
    )

    # Get the predictions and metrics. Denoised models over original data.
    pred_over_orig, metric_over_orig = xai_benchmark_denoised.predict(
        X_test,
        y_test,
        n_periods=len(y_test),
        get_metrics=True,
        rolling_forcast=False
    )

    # Get the predictions and metrics. orig models over denoised data.
    orig_over_denoised_pred, orig_over_denoised_metrics = xai_benchmark_orig.predict(
        X_test_denoised,
        y_test_denoised,
        n_periods=len(y_test_denoised),
        get_metrics=True,
        rolling_forcast=False
    )

    predictions_dict['denoised'] = {}
    metrics_dict['denoised'] = {}
    predictions_dict['denoised']['denoised_over_denoised'] = pred_over_denoised
    metrics_dict['denoised']['denoised_over_denoised'] = metric_over_denoised
    predictions_dict['denoised']['denoised_over_orig'] = pred_over_orig
    metrics_dict['denoised']['denoised_over_orig'] = metric_over_orig
    predictions_dict['denoised']['orig_over_denoised'] = orig_over_denoised_pred
    metrics_dict['denoised']['orig_over_denoised'] = orig_over_denoised_metrics

    # Correlation diff metrics
    metrics_dict['denoised']['corr_diff_orig_denoised'] = np.abs(
        orig_corr - denoised_corr
    ).values.mean()

    ## Calculate denoised histograms and Kullback-Leibler ##
    ## divergence with original and orig histograms ##
    ## ------------------------------------------------------------------------------------------ ##
    histogram_denoised = {}
    for col in df_denoised.columns:
        n_bin = histo_bins_orig[col]
        hist, _ = np.histogram(df_denoised[col], bins=n_bin, density=True)
        histogram_denoised[col] = hist + 1e-10

        kl_div = entropy(histogram_orig[col], histogram_denoised[col])
        metrics_dict['denoised'][f'kl_orig_denoised_{col}'] = kl_div

    ## Save the metrics and the predictions calculated over the entire script ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = dictionary_arrays_to_list(predictions_dict)

    # Save predictions
    with open(
        os.path.join(OUT_PATH, f'{SUBFIX_NAME}_predictions.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(predictions_dict, file, ensure_ascii=False, indent=4)

    metrics_dict = dictionary_arrays_to_list(metrics_dict)

    # Save metrics
    with open(
        os.path.join(OUT_PATH, f'{SUBFIX_NAME}_metrics.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(metrics_dict, file, ensure_ascii=False, indent=4)
