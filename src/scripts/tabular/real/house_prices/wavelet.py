# pylint: disable=import-error
# pylint: disable=wrong-import-position
"""
title: real_house_prices_exp
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a real experiment with a house_prices dataset.
The experiment consists of generating a house_prices dataset with a polinomial function and
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pywt

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
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'tabular', 'real', 'house_prices')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', 'tabular', 'real', 'house_prices')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'tabular', 'real', 'house_prices')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Show info on the terminal about how the execution is going.
VERBOSE = True
# Even if there is a checkpoint, the model is retrained.
FORCE_TRAINING = True
# Name of this experiment that will appear in the result files.
SUBFIX_NAME = 'wave'

# Local libraries
from models import XAI_benchmark

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
            'clean.parquet'
        )
    )

    # scale the data
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

    # assert there is no more categorical variables
    columnas_categoricas = df_data.select_dtypes(include=['object', 'category']).columns
    assert not list(columnas_categoricas)

    # divide the data into train/test datasets
    input_vars = list(set(df_data.columns) - set(['y']))
    X_train, X_test, y_train, y_test = train_test_split(
        df_data[input_vars].values, df_data['y'].values, test_size=0.2, random_state=42
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
    }
    xai_benchmark_orig = XAI_benchmark(
        is_ts = False,
        model_params = model_params,
        verbose = VERBOSE
    )
    xai_benchmark_orig.fit(X_train, y_train)
    orig_pred, orig_metrics = xai_benchmark_orig.predict(
        X_test,
        y_test,
        get_metrics=True
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

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        df_data[input_vars].values, df_data['y'].values, test_size=0.2, random_state=42
    )

    ## Denoise the data using Wavelet Transform decomposition ##
    ## ------------------------------------------------------------------------------------------ ##
    # Configuración
    wavelet = 'db4'  # Wavelet Daubechies 4
    df_denoised = pd.DataFrame(columns=df_data.columns)
    for col in df_data.columns:
        # Decompose the signal
        coeffs = pywt.wavedec(df_data[col], wavelet, mode='smooth')

        # Delete the coefficients below a threshold
        ## Sigma is not supposed to be known, but we can estimate it
        epsilon = 1e-10
        sigma_for_wavelet = (np.median(np.abs(coeffs[-1])) + epsilon) / 0.6745
        ## Define threshold by the universal Donoho rule
        threshold = sigma_for_wavelet * np.sqrt(2 * np.log(len(df_data[col])))
        coeffs_denoised = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]

        # Recompose the signal with the last coefficients
        df_denoised[col] = pywt.waverec(coeffs_denoised, wavelet)[:len(df_data[col])]
    df_denoised = df_denoised.copy()

    # Show the metrics
    gt_values = df_data.values
    predicted_values = df_denoised.values
    mae = mean_absolute_error(gt_values, predicted_values)
    mape = mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    nn_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'R2': r_squared
    }

    predictions_dict['orig'][SUBFIX_NAME] = predicted_values
    metrics_dict['orig'][SUBFIX_NAME] = nn_metrics

    denoised_corr = df_denoised.corr()

    ## Perform XAI benchmark over Denoised data ##
    ## -------------------------------------------------------------------------------------- ##
    xai_benchmark_denoised = XAI_benchmark(
        is_ts = False,
        model_params = model_params,
        verbose = VERBOSE
    )
    xai_benchmark_denoised.fit(df_denoised[input_vars].values, df_denoised['y'].values)
    xai_benchmark_denoised.save(
        path = os.path.join(CHECKPOINT_PATH, 'denoised'),
        subfix = f'{SUBFIX_NAME}_denoised'
    )

    # Get the predictions and metrics. Denoised models over denoised data.
    pred_over_denoised, metric_over_denoised = xai_benchmark_denoised.predict(
        df_denoised[input_vars].values,
        df_denoised['y'].values.reshape(-1,1),
        get_metrics=True
    )
    # Get the predictions and metrics. Denoised models over original data.
    pred_over_orig, metric_over_orig = xai_benchmark_denoised.predict(
        df_data[input_vars].values,
        df_data['y'].values.reshape(-1,1),
        get_metrics=True
    )
    # Get the predictions and metrics. orig models over denoised data.
    orig_over_denoised_pred, orig_over_denoised_metrics = xai_benchmark_orig.predict(
        df_denoised[input_vars].values,
        df_denoised['y'].values.reshape(-1,1),
        get_metrics=True
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
    metrics_dict['denoised']['corr_diff_orig_denoised'] = np.nanmean(np.abs(
        orig_corr - denoised_corr
    ).values)


    ## Calculate denoised histograms and Kullback-Leibler ##
    ## divergence with original and orig histograms ##
    ## -------------------------------------------------------------------------------------- ##
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
