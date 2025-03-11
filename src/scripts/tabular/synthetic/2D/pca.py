# pylint: disable=import-error
# pylint: disable=wrong-import-position
"""
title: synthetic_2D_exp
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a synthetic experiment with a 2D dataset.
The experiment consists of generating a 2D dataset with a polinomial function and
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
from sklearn.decomposition import PCA
from scipy.stats import entropy

# Seed
random.seed(42)
np.random.seed(42)

# Global variables
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'data')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints')
OUT_PATH = os.path.join(CURRENT_DIR, 'out')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Show info on the terminal about how the execution is going.
VERBOSE = False
# Even if there is a checkpoint, the model is retrained.
FORCE_TRAINING = False

# Local libraries
from utils import add_gaussian_noise
from models import XAI_benchmark

# Functions definition #
# ------------------------------------------------------------------------------------------------ #
def polinomial_function(x_var:float):
    """
    Function that takes in a value x_var and returns its polinomial value.

    Args:
        x_var (float): value to be transformed.

    Returns:
        float: result of the polinomial function.
    """

    return x_var**4 -x_var**3 -20*(x_var**2) -20*x_var +6


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
    ## Generate data based on a polynomial function ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = {}
    metrics_dict = {}

    X = np.linspace(-5, 5, 10001)
    y = polinomial_function(X)

    df_data = pd.DataFrame({'x': X, 'y': y})
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data), columns=['x', 'y'])
    X_train, X_test, y_train, y_test = train_test_split(
        df_data['x'].values, df_data['y'].values, test_size=0.2, random_state=42
    )

    ## Perform XAI benchmark over no noisy (or original) data ##
    ## ------------------------------------------------------------------------------------------ ##
    model_params = {
        'ridge': {"alpha": 1.0},
        'pls': {"n_components": 1},
        'tree': {"max_depth": 5},
        'svm': {"dual": 'auto'},
        'knn': {
            "n_neighbors": 5,
            "weights": 'uniform',
            "algorithm": 'auto',
            "leaf_size": 30,
            "p": 2,
            "n_jobs": None
        },
    }
    xai_benchmark_orig = XAI_benchmark(is_ts = False,
    model_params = model_params,
    verbose = VERBOSE
)
    xai_benchmark_orig.fit(X_train.reshape(-1,1), y_train)
    no_noise_pred, no_noise_metrics = xai_benchmark_orig.predict(
        X_test.reshape(-1,1),
        y_test, get_metrics=True
    )

    xai_benchmark_orig.save(
        path = os.path.join(CHECKPOINT_PATH,'tabular','synthetic','2D', 'no_noise'),
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
    for sigma in np.arange(0.01, 0.17, 0.01):
        sigma = round(sigma, 2)
        if sigma == 0.16:
            sigma = 'mix'
        print('\n')
        print(f'» Ruido gaussiano aplicado a los datos con sigma={sigma}')

        df_noisy = pd.DataFrame()
        if sigma != 'mix':
            df_noisy = add_gaussian_noise(
                df=df_data.copy(),
                columns=list(df_data.columns),
                mean=0.0,
                std=sigma
            )
        else:
            for s in np.arange(0.01, 0.16, 0.01):
                df_noisy = pd.concat([
                    df_noisy,
                    add_gaussian_noise(
                        df=df_data.copy(),
                        columns=list(df_data.columns),
                        mean=0.0,
                        std=s
                    )
                ])

        noisy_corr = df_noisy.corr()

        X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
            df_noisy['x'].values, df_noisy['y'].values, test_size=0.2, random_state=42
        )

        ## Calculate noisy histograms and Kullback-Leibler divergence with original histograms ##
        ## -------------------------------------------------------------------------------------- ##
        histogram_noisy = {}
        metrics_dict[sigma] = {}
        for col in df_noisy.columns:
            n_bin = histo_bins_no_noise[col]
            hist, _ = np.histogram(df_noisy[col], bins=n_bin, density=True)
            histogram_noisy[col] = hist + 1e-10

        ## Perform XAI benchmark over Noisy (with 'sigma' level noise) data ##
        ## -------------------------------------------------------------------------------------- ##
        xai_benchmark_noisy = XAI_benchmark(
            is_ts = False,
            model_params = model_params,
            verbose = VERBOSE
        )
        xai_benchmark_noisy.fit(X_train_noisy.reshape(-1,1), y_train_noisy)
        pred, metrics = xai_benchmark_noisy.predict(
            X_test_noisy.reshape(-1,1),
            y_test_noisy,
            get_metrics=True
        )
        xai_benchmark_noisy.save(
            path = os.path.join(CHECKPOINT_PATH,'tabular','synthetic','2D', f'{sigma}'),
            subfix = f'noise_{sigma}'
        )
        predictions_dict[sigma] = pred
        metrics_dict[sigma] = metrics

        ## Denoise the data using Principal Components Analysis ##
        ## -------------------------------------------------------------------------------------- ##
        pca = PCA()
        pca.fit(df_noisy)

        # Select principal components with sufficient variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        # 95% threshold for explained variance
        n_components = np.argmax(cumulative_variance >= 0.95) + 1

        # Reduce dimensionality and reconstruct the signal
        pca_denoising = PCA(n_components=n_components)
        data_reduced = pca_denoising.fit_transform(df_noisy)
        df_denoised = pca_denoising.inverse_transform(data_reduced)
        df_denoised = pd.DataFrame(df_denoised, columns=df_noisy.columns)

        # Calc the metrics
        gt_values = df_data.values
        if sigma == 'mix':
            gt_values = np.tile(gt_values, (15,1))
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

        predictions_dict[sigma]['pca_transform'] = predicted_values
        metrics_dict[sigma]['pca_transform'] = nn_metrics

        denoised_corr = df_denoised.corr()

        ## Perform XAI benchmark over Denoised data ##
        ## -------------------------------------------------------------------------------------- ##
        xai_benchmark_denoised = XAI_benchmark(is_ts = False,
        model_params = model_params,
        verbose = VERBOSE
    )
        xai_benchmark_denoised.fit(df_denoised['x'].values.reshape(-1,1), df_denoised['y'].values)
        xai_benchmark_denoised.save(
            path = os.path.join(CHECKPOINT_PATH,'tabular','synthetic','2D', f'{sigma}'),
            subfix = f'ma_denoised_{sigma}'
        )

        # Get the predictions and metrics. Denoised models over denoised data.
        pred_over_denoised, metric_over_denoised = xai_benchmark_denoised.predict(
            df_denoised['x'].values.reshape(-1,1),
            df_denoised['y'].values.reshape(-1,1),
            get_metrics=True
        )
        # Get the predictions and metrics. Denoised models over no noise (original) data.
        pred_over_orig, metric_over_orig = xai_benchmark_denoised.predict(
            df_data['x'].values.reshape(-1,1),
            df_data['y'].values,
            get_metrics=True
        )
        # Get the predictions and metrics. Noisy models over denoised data.
        noisy_over_denoised_pred, noisy_over_denoised_metrics = xai_benchmark_noisy.predict(
            df_denoised['x'].values.reshape(-1,1),
            df_denoised['y'].values.reshape(-1,1),
            get_metrics=True
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
        os.path.join(OUT_PATH, 'tabular', 'synthetic', '2D', 'pca_predictions.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(predictions_dict, file, ensure_ascii=False, indent=4)

    metrics_dict = dictionary_arrays_to_list(metrics_dict)
    # Save metrics
    with open(
        os.path.join(OUT_PATH, 'tabular', 'synthetic', '2D', 'pca_metrics.json'),
        'w',
        encoding='utf-8') as file:
        json.dump(metrics_dict, file, ensure_ascii=False, indent=4)
