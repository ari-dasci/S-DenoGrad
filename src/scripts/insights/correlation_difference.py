"""
This script analyzes correlation differences across datasets and denoising methods, 
generating grouped bar plots for visualization. It processes data from a specified 
directory structure, extracts correlation difference metrics, and saves the plots 
to an output directory.

Functions:
----------
- list_files(init_folder: str) -> list:
    Recursively lists all files in the given directory and its subdirectories 
    with relative paths.

- validate_folder(selected_folder: str):
    Validates that the selected path is a folder. Raises a ValueError if the 
    selected path is a file.

- get_correlation_data(dataset_path: str, methods_dict: dict, sigma: str='') -> dict:
    Extracts correlation difference data for a given dataset and maps models 
    to denoising methods.

- plot_and_save_correlation(df_corr: pd.DataFrame, data_type: str, data_origin: str, 
                             dataset: str, out_path: str):

- main():
    Main function to analyze correlation differences across datasets and 
    denoising methods. It validates the folder structure, processes datasets, 
    and generates plots.

Global Variables:
-----------------
- _CURRENT_DIR: str
    The current working directory of the script.

- DATA_PATH: str
    Path to the input data directory.

- OUT_PATH: str
    Path to the output directory for saving plots.

- methods_dict: dict
    A mapping of model names to denoising method names.

Usage:
------
Run the script directly to process datasets and generate correlation difference 
plots. Ensure the input data directory structure matches the expected format.
"""
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# Libs
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Local imports
sys.path.append(os.getcwd())
from src.libs.utils import make_dir


# Global vars
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'out')
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'insights', 'correlation')


def list_files(init_folder: str):
    """
    Recursively lists all files in the given directory and its subdirectories with relative paths.

    Parameters:
    init_folder (str): The initial directory path.

    Returns:
    list: A list of relative file paths.
    """
    archivos = []
    for root, _, files in os.walk(init_folder):
        for file in files:
            ruta_absoluta = os.path.join(root, file)
            archivos.append(ruta_absoluta)
    return archivos


def validate_folder(selected_folder: str):
    """
    Validates that the selected path is a folder.
    Raises:
        ValueError: If the selected path is a file instead of a folder.
    """
    if os.path.isfile(selected_folder):
        raise ValueError('A folder must be selected.')


def get_correlation_data(dataset_path: str, methods_dict: dict, sigma: str=''):
    """
    Extracts correlation difference data for a given dataset.

    Parameters:
    dataset_path (str): Path to the dataset folder.
    methods_dict (dict): Mapping of model names to denoising method names.

    Returns:
    dict: A dictionary containing correlation difference data for each method.
    """
    correlation_dict_to_plot = {}
    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        # Look for the metrics file
        metrics_file = next(
            (f for f in os.listdir(model_path) if 'metrics' in f and sigma in f), None
        )
        if not metrics_file:
            continue

        with open(os.path.join(model_path, metrics_file), 'r', encoding='utf-8') as f:
            file_metrics = json.load(f)

        # Extract correlations_diff values
        method = methods_dict.get(model, model)
        read_corr_diff_dict = file_metrics.get('correlations_diff', {})
        correlation_dict_to_plot[method] = {
            key: value for key, value in read_corr_diff_dict.items()
            if value is not None
        }

    return correlation_dict_to_plot

def plot_and_save_correlation(df_corr, data_type, data_origin, dataset, out_path):
    """
    Creates and saves a grouped bar plot for correlation differences.

    Parameters:
    df_corr (pd.DataFrame): DataFrame containing correlation difference data.
    data_type (str): Type of data (e.g., 'tabular', 'time_series').
    data_origin (str): Origin of data (e.g., 'real', 'synthetic').
    dataset (str): Dataset name.
    out_path (str): Path to save the output plot.
    """
    if df_corr.empty:
        return

    # Transpose the DataFrame to group bars by model
    df_corr = df_corr.T
    df_corr.index.name = 'Denoising method'
    df_corr.reset_index(inplace=True)
    df_melted = df_corr.melt(id_vars='Denoising method',
                                var_name='Correlation difference type',
                                value_name='Correlation difference')

    # Create and save the grouped bar plot
    _, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=df_melted,
        x='Denoising method',
        y='Correlation difference',
        hue='Correlation difference type',
        ax=ax,
        palette='ch:start=.5,rot=-0.6,dark=.5,light=.8'#'Spectral'# 'YlOrBr'# 'viridis'
    )

    # Display the value of each column above or inside the bars
    y_max = ax.get_ylim()[1]
    for container in ax.containers:
        for bar_var in container:
            height = bar_var.get_height()
            text = f'{height:.4f}'
            if height > 0.15 * y_max:
                ax.text(
                    bar_var.get_x() + bar_var.get_width() / 2,
                    height / 2,
                    text,
                    ha='center',
                    va='center',
                    fontsize=10,
                    # color='white',
                    rotation=90
                )
            else:
                ax.text(
                    bar_var.get_x() + bar_var.get_width() / 2,
                    height + (0.02 * y_max),
                    text,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    rotation=90
                )

    # Configure plot title and labels
    data_str = f'{data_type} {data_origin} {dataset.upper()}'
    ax.set_title(f"Correlation difference comparison\nData: {data_str}", fontsize=14)
    ax.set_ylabel("Correlation difference", fontsize=12)
    ax.set_xlabel("Denoising method", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title='Correlation difference type', fontsize=10, title_fontsize=12)

    # Save the figure
    fig_path = os.path.join(out_path, data_type, data_origin)
    make_dir(fig_path)
    fig_path = os.path.join(fig_path, f'{dataset}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Main function to analyze correlation differences across datasets and denoising methods.
    """
    # Folder or file selection
    selected_folder = DATA_PATH

    # Validate the folder
    validate_folder(selected_folder)

    methods_dict = {
        'dae': 'DAE',
        'dlnr': 'DLNR',
        'emd': 'EMD',
        'kalman_filter': 'Kalman',
        'moving_average': 'MA',
        'pca': 'PCA',
        'resnet': 'ResNet',
        'wavelet_transform': 'Wavelet'
    }

    # Walk through the folder structure
    for data_type in ['tabular', 'time_series']:
        data_type_path = os.path.join(selected_folder, data_type)
        if not os.path.exists(data_type_path):
            continue

        for data_origin in ['real', 'synthetic']:
            origin_path = os.path.join(data_type_path, data_origin)
            if not os.path.exists(origin_path):
                continue

            for dataset in os.listdir(origin_path):
                dataset_path = os.path.join(origin_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                # Extract correlation data
                sigma = ''
                if data_origin == 'synthetic':
                    sigma = '0.05'
                correlation_dict_to_plot = get_correlation_data(dataset_path, methods_dict, sigma)

                # Create DataFrame for plotting
                df_corr = pd.DataFrame(correlation_dict_to_plot)

                # Plot and save the correlation differences
                plot_and_save_correlation(df_corr, data_type, data_origin, dataset, OUT_PATH)

# Main
if __name__ == "__main__":
    main()
