"""
This script processes KL divergence metrics for various models applied to datasets, 
generating visualizations (heatmaps and bar plots) to analyze the results. It supports 
both real and synthetic data, with functionality to handle multiple data types and noise levels.

Modules:
    - os: Provides functions for interacting with the operating system.
    - sys: Provides access to system-specific parameters and functions.
    - json: Handles JSON file reading and writing.
    - matplotlib.pyplot: Used for creating visualizations.
    - pandas: Provides data manipulation and analysis tools.
    - seaborn: Used for advanced data visualization.
    - src.libs.utils: Contains utility functions like `make_dir`.

Global Variables:
    - _CURRENT_DIR: The current working directory.
    - _FOLDERS: List of folder names in the current directory path.
    - _PROJECT_FOLDER_INDEX: Index of the project folder in the directory path.
    - DATA_PATH: Path to the output data directory.
    - OUT_PATH: Path to the output insights directory for KL divergence.
    - methods_dict: Dictionary mapping method names to their descriptive labels.

Functions:
    - load_metrics(model_path, metrics_file, is_real):
        Loads KL divergence metrics from a JSON file for a given model.

    - create_heatmap(df_kl, dataset, data_type, data_origin, noise_lvl, is_real):

    - create_barplot(df_kl, dataset, data_type, data_origin, noise_lvl, is_real):

    - process_dataset(dataset_path, dataset, data_type, data_origin, noise_lvl, is_real):
        Processes a dataset by computing KL divergence metrics and generating visualizations.

    - process_data_type(data_type, noise_lvl):
        real and synthetic data origins.

    - main():
        Main function to process data types with a specified noise level, iterating over 
        predefined data types and invoking the processing functions.

Usage:
    Run the script directly to process KL divergence metrics for predefined data types 
    ('tabular', 'time_series') and a fixed noise level (0.05). The script generates 
    heatmaps and bar plots for each dataset and saves them in the output directory.
"""
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# Libs
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Local imports
sys.path.append(os.getcwd())
from src.libs.utils import make_dir

# Global vars
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'out')
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'insights', 'kl_divergence')

methods_dict = {
    'dae': 'DAE',
    'dlnr': 'DenoGrad',
    'emd': 'EMD',
    'kalman_filter': 'Kalman',
    'moving_average': 'MA',
    'pca': 'PCA',
    'resnet': 'ResNet',
    'wavelet_transform': 'Wavelet'
}

def load_metrics(model_path, metrics_file, is_real):
    """
    Load and retrieve KL divergence metrics from a JSON file.
    Args:
        model_path (str): The path to the directory containing the metrics file.
        metrics_file (str): The name of the JSON file containing the metrics.
        is_real (bool): A flag indicating whether to retrieve metrics for real data.
    Returns:
        dict: A dictionary containing the 'noisy_denosied' KL divergence metrics.
              Returns an empty dictionary if the key is not found.
    """
    with open(os.path.join(model_path, metrics_file), 'r', encoding='utf-8') as f:
        file_metrics = json.load(f)

    if is_real:
        return file_metrics.get('KL_divergence', {}).get('noisy_denosied', {})
    return file_metrics.get('KL_divergence', {}).get('noisy_denosied', {})

def create_heatmap(df_kl, dataset, data_type, data_origin, noise_lvl, is_real):
    """
    Generates and saves a heatmap visualization of KL divergence values.
    Parameters:
        df_kl (pd.DataFrame): A DataFrame containing KL divergence values, 
                              where rows represent methods and columns represent variables.
        dataset (str): The name of the dataset being analyzed.
        data_type (str): The type of data (e.g., "train", "test").
        data_origin (str): The origin of the data (e.g., "synthetic", "real").
        noise_lvl (float): The noise level applied to the data (used for synthetic data).
        is_real (bool): A flag indicating whether the data is real (True) or synthetic (False).
    Raises:
        ValueError: If the input DataFrame `df_kl` is empty or has no columns.
    Side Effects:
        - Saves the generated heatmap as a PNG file in a directory structure based on the 
          `data_type`, `data_origin`, and `dataset` parameters.
        - Creates necessary directories if they do not exist.
    Notes:
        - The heatmap uses the "coolwarm" colormap and includes gridlines with a linewidth of 0.01.
        - The title of the heatmap includes information about the dataset, data type, data origin, 
          and noise level (if applicable).
    """
    _, ax = plt.subplots(figsize=(10, 6))
    if df_kl.empty or df_kl.shape[1] == 0:
        raise ValueError(f"Dataset: {dataset}. Empty DataFrame.")

    sns.heatmap(df_kl, cmap="coolwarm", linewidths=0.01)

    dataset_str = dataset.upper()
    noise_str = f'_s{noise_lvl}' if not is_real else ''
    data_str = f'{data_type} {data_origin} {dataset_str}{noise_str}'

    ax.set_title(
        f"KL divergence per Variable and Method\nData: {data_str}",
        fontsize=14
    )
    ax.set_ylabel("Variables", fontsize=12)
    ax.set_xlabel("Methods", fontsize=12)

    fig_path = os.path.join(OUT_PATH, data_type, data_origin, dataset)
    make_dir(fig_path)
    fig_path = os.path.join(fig_path, f'{dataset}_heatmap.png')
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def create_barplot(df_kl, dataset, data_type, data_origin, noise_lvl, is_real):
    """
    Creates and saves a bar plot visualizing the mean KL divergence per method.
    Parameters:
        df_kl (pd.DataFrame): DataFrame containing KL divergence values for different methods.
                              The columns represent methods, and the rows represent observations.
        dataset (str): Name of the dataset being analyzed.
        data_type (str): Type of data (e.g., "train", "test").
        data_origin (str): Origin of the data (e.g., "synthetic", "real").
        noise_lvl (float): Noise level applied to the data (used in the plot title and file naming).
        is_real (bool): Indicates whether the data is real or synthetic.
    Raises:
        ValueError: If the input DataFrame `df_kl` is empty or has no columns.
    Saves:
        A bar plot image file in the specified output directory. The file is saved with the name
        format `<dataset>_barplot.png` under a directory structure based on `data_type`,
        `data_origin`, and `dataset`.
    Notes:
        - The function uses a dictionary `methods_dict` to map column names to more descriptive
            labels.
        - The bar plot includes mean KL divergence values for each method, with values displayed as
            labels.
        - The output directory is created if it does not already exist.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    if df_kl.empty or df_kl.shape[1] == 0:
        raise ValueError("Empty DataFrame for bar plot.")

    df_kl.columns = [methods_dict.get(col, col) for col in df_kl.columns]

    sns.barplot(
        x=df_kl.mean().index,
        y=df_kl.mean().values,
        ax=ax,
        palette='viridis',
        hue=df_kl.mean().index,
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8, padding=3)

    dataset_str = dataset.upper()
    noise_str = f'_s{noise_lvl}' if not is_real else ''
    data_str = f'{data_type} {data_origin} {dataset_str}{noise_str}'

    ax.set_title(
        f"Mean KL divergence per Method\nData: {data_str}",
        fontsize=14
    )
    ax.set_ylabel("KL mean", fontsize=12)
    ax.set_xlabel("Methods", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig_path = os.path.join(OUT_PATH, data_type, data_origin, dataset)
    make_dir(fig_path)
    fig_path = os.path.join(fig_path, f'{dataset}_barplot.png')
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def process_dataset(dataset_path, dataset, data_type, data_origin, noise_lvl, is_real):
    """
    Processes a dataset by computing KL divergence metrics for models in the dataset path
    and generating visualizations (heatmap and barplot) based on the results.
    Args:
        dataset_path (str): Path to the directory containing model subdirectories.
        dataset (str): Name of the dataset being processed.
        data_type (str): Type of data (e.g., "image", "text").
        data_origin (str): Origin of the data (e.g., "synthetic", "real").
        noise_lvl (int or str): Noise level to filter the metrics files.
        is_real (bool): Flag indicating whether the dataset is real or synthetic.
    Returns:
        None
    """
    kl_orig_denoised_dict = {}

    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        metrics_file = next(
            (
                f for f in os.listdir(model_path)
                if 'metrics' in f and (str(noise_lvl) in f or is_real)
            ), None
        )

        if not metrics_file:
            continue

        kl_metrics = load_metrics(model_path, metrics_file, is_real)
        kl_orig_denoised_dict[model] = kl_metrics

    df_kl = pd.DataFrame(kl_orig_denoised_dict)
    df_kl = df_kl[sorted(df_kl.columns)]

    df_kl.columns = [col.upper().split('_')[0] for col in df_kl.columns]
    df_kl.rename(columns={'DLNR': 'DenoGrad', 'MOVING': 'MA'}, inplace=True)

    create_heatmap(df_kl, dataset, data_type, data_origin, noise_lvl, is_real)
    create_barplot(df_kl, dataset, data_type, data_origin, noise_lvl, is_real)

def process_data_type(data_type, noise_lvl):
    """
    Processes datasets of a specific data type and noise level by iterating through
    real and synthetic data origins and invoking the `process_dataset` function.
    Args:
        data_type (str): The type of data to process (e.g., 'images', 'text').
        noise_lvl (float): The noise level to consider during processing.
    Returns:
        None: The function performs processing and does not return a value.
    Notes:
        - The function expects a predefined `DATA_PATH` variable to construct paths.
        - If the `data_type` directory does not exist, the function exits early.
        - For each dataset in the 'real' and 'synthetic' subdirectories, the function
          checks if the dataset path is a directory before processing it.
        - The `process_dataset` function is called with the dataset path and metadata
          including whether the data origin is 'real'.
    """
    data_type_path = os.path.join(DATA_PATH, data_type)
    if not os.path.exists(data_type_path):
        return

    for data_origin in ['real', 'synthetic']:
        origin_path = os.path.join(data_type_path, data_origin)
        if not os.path.exists(origin_path):
            continue

        for dataset in os.listdir(origin_path):
            dataset_path = os.path.join(origin_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            is_real = data_origin == 'real'
            process_dataset(dataset_path, dataset, data_type, data_origin, noise_lvl, is_real)

def main():
    """
    Main function to process data types with a specified noise level.
    This function iterates over a predefined list of data types and processes
    each one using the `process_data_type` function, applying a fixed noise level.
    Args:
        None
    Returns:
        None
    """
    noise_lvl = 0.05
    for data_type in ['tabular', 'time_series']:
        process_data_type(data_type, noise_lvl)

if __name__ == "__main__":
    main()
