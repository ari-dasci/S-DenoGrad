# pylint: disable=import-error
# pylint: disable=wrong-import-position
"""
title: synthetic_time_series_exp
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a synthetic experiment with a time_series dataset.
The experiment consists of generating a time_series dataset with a polinomial function and
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'time_series', 'synthetic')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', 'time_series', 'synthetic')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'time_series', 'synthetic')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Show info on the terminal about how the execution is going.
VERBOSE = True
# Even if there is a checkpoint, the model is retrained.
FORCE_TRAINING_PRE_XAI = True
FORCE_TRAINING_NOISY_XAI = True
FORCE_TRAINING_NN = True
FORCE_TRAINING_POST_XAI = True
# Name of this experiment that will appear in the result files.
SUBFIX_NAME = 'gradient'
IS_TS = True

# Local libraries
from utils import add_gaussian_noise, symmetric_mean_absolute_percentage_error
from dataset import SlidingWindowDataset
from models import Trainer, XAI_benchmark, LSTMModel
from dlnr import DLNoiseReduction

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
    elif isinstance(array_d, (np.float32, np.float64)):
        return float(array_d)
    else:
        return array_d


# Main #
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
    ## Load data ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = {}
    metrics_dict = {}
    df_data = pd.read_parquet(
        os.path.join(
            DATA_PATH,
            '1000s_5v_24w',
            'clean.parquet'
        )
    )
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)

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

    ## Perform XAI benchmark over no noisy (or original) data ##
    ## ------------------------------------------------------------------------------------------ ##
    model_params = {
        'ridge': {"alpha": 1.0},
        # 'pls': {"n_components": 1},
        'pls': None,
        'tree': {"max_depth": 5},
        # 'svm': {"kernel": 'poly', "degree": 2},
        'svm': None,
        'knn': {
            "n_neighbors": 5,
            "weights": 'uniform',
            "algorithm": 'auto',
            "leaf_size": 30,
            "p": 2,
            "n_jobs": None
        },
        # 'arima': {
        #     'order': (1, 1, 0),
        #     'seasonal_order': (4, 0, 5, 12)
        # },
        'arima': None,
        'auto_arima': None
    }
    xai_benchmark_orig = XAI_benchmark(
        is_ts = IS_TS,
        model_params = model_params,
        verbose = VERBOSE
    )

    if FORCE_TRAINING_PRE_XAI:
        xai_benchmark_orig.fit(X_train, y_train)
    else:
        xai_benchmark_orig.load(
            folder_path=os.path.join(CHECKPOINT_PATH, 'no_noise')
        )

    no_noise_pred, no_noise_metrics = xai_benchmark_orig.predict(
        X_test,
        y_test,
        n_periods=len(y_test),
        get_metrics=True,
        rolling_forcast=False
    )

    xai_benchmark_orig.save(
        path = os.path.join(CHECKPOINT_PATH, 'no_noise'),
        subfix = 'no_noise'
    )

    predictions_dict['no_noise'] = no_noise_pred
    metrics_dict['no_noise'] = no_noise_metrics

    ## Calculate orignal histograms and correlation matrix ##
    ## ------------------------------------------------------------------------------------------ ##
    no_noise_corr = df_data.corr()
    histogram_no_noise = {}
    histo_bins_no_noise = {}
    for col in df_data.columns:
        hist, bin_edges = np.histogram(df_data[col], bins='auto', density=True)
        histogram_no_noise[col] = hist + 1e-10
        histo_bins_no_noise[col] = len(bin_edges) - 1

    ## Add gaussian noise to the data in all variables ##
    ## ------------------------------------------------------------------------------------------ ##
    # for sigma in np.arange(0.01, 0.16, 0.01):
    for sigma in [0.05]:
        sigma = round(sigma, 2)

        if VERBOSE:
            print('\n')
            print(f'» Ruido gaussiano aplicado a los datos con sigma={sigma}')

        df_noisy = pd.DataFrame()
        df_noisy = add_gaussian_noise(
            data=df_data.copy(),
            columns=list(df_data.columns),
            mean=0.0,
            std=sigma
        )

        noisy_corr = df_noisy.corr()

        # Desplazar la última columna hacia arriba
        df_noisy['y_shifted'] = df_noisy['y'].shift(-1)
        # Eliminar la última fila porque tendrá un NaN en la última columna
        df_noisy = df_noisy.dropna().reset_index(drop=True)
        y_shifted_noisy = df_noisy['y_shifted'].copy()
        df_noisy = df_noisy.drop(columns=['y_shifted'])

        # divide the data into train/test datasets
        input_vars = df_noisy.columns
        X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
            df_noisy[input_vars].values, y_shifted_noisy, test_size=0.2, shuffle=False
        )

        ## Calculate noisy histograms and Kullback-Leibler divergence with original histograms ##
        ## -------------------------------------------------------------------------------------- ##
        histogram_noisy = {}
        metrics_dict[sigma] = {}
        for col in df_noisy.columns:
            n_bin = histo_bins_no_noise[col]
            hist, _ = np.histogram(df_noisy[col], bins=n_bin, density=True)
            histogram_noisy[col] = hist + 1e-10

            kl_div = entropy(histogram_no_noise[col], histogram_noisy[col])
            metrics_dict[sigma][f'kl_no_noise_noisy_{col}'] = kl_div


        ## Perform XAI benchmark over Noisy (with 'sigma' level noise) data ##
        ## -------------------------------------------------------------------------------------- ##
        xai_benchmark_noisy = XAI_benchmark(
            is_ts = IS_TS,
            model_params = model_params,
            verbose = VERBOSE
        )
        if FORCE_TRAINING_NOISY_XAI:
            xai_benchmark_noisy.fit(X_train_noisy, y_train_noisy)
        else:
            xai_benchmark_noisy.load(os.path.join(CHECKPOINT_PATH, 'noisy', f'{sigma}'))

        pred, metrics = xai_benchmark_noisy.predict(
            X_test_noisy,
            y_test_noisy,
            n_periods=len(y_test_noisy),
            get_metrics=True,
            rolling_forcast=False
        )
        xai_benchmark_noisy.save(
            path = os.path.join(CHECKPOINT_PATH, 'noisy', f'{sigma}')
        )
        predictions_dict[sigma] = pred
        metrics_dict[sigma] = metrics

        ## Declare a Neural Network model and prepare the data to train it ##
        ## -------------------------------------------------------------------------------------- ##

        ### 'y' parameter should be the same as 'x' in a normal problem where the original ###
        ### (no noise/clean) data is not available. Here, the original data is used. ###
        ## -------------------------------------------------------------------------------------- ##

        # Create the dataloaders
        # divide the data into train/test datasets
        input_vars = df_noisy.columns
        X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
            df_noisy[input_vars].values, df_noisy['y'].values, test_size=0.2, shuffle=False
        )
        batch_size = 64
        window_size = 30
        train_dataset = SlidingWindowDataset(
            X_train_nn,
            y_train_nn,
            window_size=window_size,
            future=1
        )
        val_dataset = SlidingWindowDataset(
            X_test_nn,
            y_test_nn,
            window_size=window_size,
            future=1
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create Neural Network model
        input_size = X_train_nn.shape[1] # Número de variables de entrada
        hidden_size = 128  # Número de neuronas en la capa oculta
        output_size = 1  # Predicción de una variable
        model = LSTMModel(input_size, hidden_size, output_size).to(device)

        # Set model parameters and create the model Trainer object
        lr = 0.001
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        denoiser_checkpoint_path = os.path.join(
            CHECKPOINT_PATH,
            f'{sigma}',
            f'{SUBFIX_NAME}.pth'
        )

        # Define the trainer
        trainer_basic = Trainer(
            model=model,
            train_generator=train_dataloader,
            val_generator=val_dataloader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch_scheduler=None,
            batch_scheduler=None,
            patience=15,
            epochs=500,
            checkpoints_path=denoiser_checkpoint_path
        )

        ## Train the Neural Network or load its weights from a checkpoint ##
        ## -------------------------------------------------------------------------------------- ##
        if os.path.exists(denoiser_checkpoint_path) and not FORCE_TRAINING_NN:
            model.load_state_dict(torch.load(denoiser_checkpoint_path))
            if VERBOSE:
                print(
                    'Checkpoint loaded for the neural network model from path: ',
                    denoiser_checkpoint_path
                )
        else:
            model, _, _, _, _ = trainer_basic.fit(verbose=VERBOSE)


        ## Predict and get the metrics for de NN model ##
        ## -------------------------------------------------------------------------------------- ##
        # y_pred_test = model(
        #     torch.tensor(X_test_nn).float().to(device)
        # ).cpu().detach().numpy().reshape(-1)

        predictions_test = trainer_basic.eval_dataloader(val_dataloader)
        y_pred_test = np.array([y for x in predictions_test for y in x])

        # Show the metrics
        gt_values = y_test_nn[window_size:]
        predicted_values = y_pred_test
        mae = mean_absolute_error(gt_values, predicted_values)
        smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
        mse = mean_squared_error(gt_values, predicted_values)
        rmse = np.sqrt(mse)
        r_squared = r2_score(gt_values, predicted_values)

        nn_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'smape': smape,
            'R2': r_squared
        }

        predictions_dict[sigma][SUBFIX_NAME] = predicted_values
        metrics_dict[sigma][SUBFIX_NAME] = nn_metrics

        ## Perform gradient-based denoising method ##
        ## -------------------------------------------------------------------------------------- ##
        x_sliding = df_noisy[input_vars].values
        y_sliding = df_noisy[['y']]
        df_to_denoise = SlidingWindowDataset(x_sliding, y_sliding, window_size=window_size, future=1)

        dlnr = DLNoiseReduction(model=model, criterion=criterion, is_ts=IS_TS)
        dlnr.fit(df_to_denoise)

        df_denoised = df_noisy.copy()
        df_denoised[input_vars], old_y, x_gradients, y_gradients = dlnr.transform(
            nrr=0.05,
            nr_threshold=0.02,
            max_epochs=1000,
            plot_progress=False,
            path_to_save_imgs=None,
            denoise_y=False
        )

        denoised_corr = df_denoised.corr()

        ## Perform XAI benchmark over Denoised data ##
        ## -------------------------------------------------------------------------------------- ##
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
            path = os.path.join(CHECKPOINT_PATH, 'denoised', f'{sigma}'),
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
        noisy_over_denoised_pred, noisy_over_denoised_metrics = xai_benchmark_noisy.predict(
            X_test_denoised,
            y_test_denoised,
            n_periods=len(y_test_denoised),
            get_metrics=True,
            rolling_forcast=False
        )

        predictions_dict[sigma]['denoised_over_denoised'] = pred_over_denoised
        metrics_dict[sigma]['denoised_over_denoised'] = metric_over_denoised
        predictions_dict[sigma]['denoised_over_orig'] = pred_over_orig
        metrics_dict[sigma]['denoised_over_orig'] = metric_over_orig
        predictions_dict[sigma]['noisy_over_denoised'] = noisy_over_denoised_pred
        metrics_dict[sigma]['noisy_over_denoised'] = noisy_over_denoised_metrics

        # Correlation diff metrics
        metrics_dict[sigma]['corr_diff_orig_noisy'] = np.nanmean(np.abs(
            no_noise_corr - noisy_corr
        ).values)
        metrics_dict[sigma]['corr_diff_orig_denoised'] = np.nanmean(np.abs(
            no_noise_corr - denoised_corr
        ).values)
        metrics_dict[sigma]['corr_diff_noisy_denoised'] = np.nanmean(np.abs(
            noisy_corr - denoised_corr
        ).values)


        ## Calculate denoised histograms and Kullback-Leibler ##
        ## divergence with original and noisy histograms ##
        ## -------------------------------------------------------------------------------------- ##
        histogram_denoised = {}
        for col in df_denoised.columns:
            n_bin = histo_bins_no_noise[col]
            hist, _ = np.histogram(df_denoised[col], bins=n_bin, density=True)
            histogram_denoised[col] = hist + 1e-10

            kl_div = entropy(histogram_no_noise[col], histogram_noisy[col])
            metrics_dict[sigma][f'kl_orig_noisy_{col}'] = kl_div

            kl_div = entropy(histogram_no_noise[col], histogram_denoised[col])
            metrics_dict[sigma][f'kl_orig_denoised_{col}'] = kl_div

            kl_div = entropy(histogram_noisy[col], histogram_denoised[col])
            metrics_dict[sigma][f'kl_noisy_denoised_{col}'] = kl_div


    ## Save the metrics and the predictions calculated over the entire script ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = dictionary_arrays_to_list(predictions_dict)

    # Save predictions
    with open(
        os.path.join(OUT_PATH, f'{SUBFIX_NAME}_predictions.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(predictions_dict, file, ensure_ascii=False, indent=4)

    # Save metrics
    metrics_dict = dictionary_arrays_to_list(metrics_dict)
    with open(
        os.path.join(OUT_PATH, f'{SUBFIX_NAME}_metrics.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(metrics_dict, file, ensure_ascii=False, indent=4)
