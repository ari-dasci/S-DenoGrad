"""
This script analyzes the denoising performance of various models across datasets and generates
visualizations of R2 scores for different scenarios. It processes both real and synthetic datasets,
extracts metrics data, and creates bar plots for each denoising method.

Modules:
    - os: Provides functions for interacting with the operating system.
    - sys: Provides access to system-specific parameters and functions.
    - json: Provides functions for working with JSON data.
    - matplotlib.pyplot: Used for creating static, interactive, and animated visualizations.
    - seaborn: A Python visualization library based on matplotlib.
    - pandas: A data analysis and manipulation library.
    - tqdm: A library for creating progress bars.
    - src.libs.utils: Contains utility functions, such as `make_dir`.

Functions:
    - list_files(init_folder): Recursively lists all files in a directory and its subdirectories.
    - get_metrics_data(dataset_path, sigma='', is_real=False): Extracts metrics data for a
        given dataset.
    - process_and_plot_metrics(metrics_dict, data_type, data_origin, dataset, out_path): Processes
        metrics data and generates bar plots for each denoising method.
    - main(): Main function to analyze denoising performance across datasets and models.

Global Variables:
    - _CURRENT_DIR: The current working directory.
    - _FOLDERS: A list of folder names in the current directory path.
    - _PROJECT_FOLDER_INDEX: The index of the project folder in the directory path.
    - DATA_PATH: Path to the output data folder.
    - OUT_PATH: Path to save the output insights and plots.

Usage:
    Run the script to analyze denoising performance and generate visualizations for datasets
    located in the specified `DATA_PATH`. The results will be saved in the `OUT_PATH` directory.

Note:
    Ensure that the required dependencies are installed and the `src.libs.utils` module is
    accessible.
"""
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# Libs
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
# Local imports
sys.path.append(os.getcwd())
from src.libs.utils import make_dir

# Global vars
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'out')
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'insights', 'r2', 'denoising_performance')


def list_files(init_folder):
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


def get_metrics_data(dataset_path: str, sigma: str = '', is_real: bool = False):
    """
    Extracts metrics data for a given dataset.

    Parameters:
    dataset_path (str): Path to the dataset folder.
    sigma (str): Noise level filter for synthetic data.
    is_real (bool): Whether the dataset is real or synthetic.

    Returns:
    dict: A dictionary containing metrics data for each model.
    """
    metrics_dict = {}
    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        # Look for the metrics file
        metrics_file = next(
            (f for f in os.listdir(model_path) if 'metrics' in f and (sigma in f or is_real)), None
        )
        if not metrics_file:
            continue

        with open(os.path.join(model_path, metrics_file), 'r', encoding='utf-8') as f:
            file_metrics = json.load(f)

        # Extract relevant metrics
        metrics_dict[model] = {}
        if is_real:
            metrics_dict[model]['noisy'] = file_metrics.get('XAI', {}).get('noisy', {})
            metrics_dict[model]['denoised'] = file_metrics.get('XAI', {}).get('denoised', {})
        else:
            metrics_dict[model]['orig'] = file_metrics.get('XAI', {}).get('orig', {})
            metrics_dict[model]['noisy'] = file_metrics.get('XAI', {}).get('noisy', {})
            metrics_dict[model]['denoised'] = file_metrics.get('XAI', {}).get('denoised', {})

    return metrics_dict


def process_and_plot_metrics(metrics_dict, data_type, data_origin, dataset, out_path):
    """
    Processes metrics data and generates separate plots for each denoising method.

    Parameters:
    metrics_dict (dict): Dictionary containing metrics data for each model.
    data_type (str): Type of data (e.g., 'tabular', 'time_series').
    data_origin (str): Origin of data (e.g., 'real', 'synthetic').
    dataset (str): Dataset name.
    out_path (str): Path to save the output plots.
    """
    progress_bar = tqdm(total=len(metrics_dict), desc=f"Creating plots for {dataset}", unit="model")

    for model, metrics in metrics_dict.items():
        # Extract R2 values for different scenarios
        dfs = []
        for _, train_test_nomenclature_metrics in metrics.items():
            if not train_test_nomenclature_metrics:
                continue  # Skip if scenario metrics are missing or empty
            # scenario_df = pd.DataFrame.from_dict(train_test_nomenclature_metrics, orient='index')
            # Extract R2 values for each train_test_comb in the nested dictionary
            scenario_r2 = {
                train_test_comb: {
                    model_name: model_metrics.get('R2')
                    for model_name, model_metrics in models_metrics.items()
                    if model_metrics.get('R2') is not None
                }
                for train_test_comb, models_metrics in train_test_nomenclature_metrics.items()
            }
            scenario_df = pd.DataFrame.from_dict(scenario_r2, orient='index').fillna(0)
            # Replace negative values with 0
            scenario_df[scenario_df < 0] = 0
            dfs.append(scenario_df)

        # Combine all scenarios into a single DataFrame
        if not dfs:
            print(f"Warning: No valid data for model '{model}'. Skipping plot.")
            exit()

        # Combine all scenarios into a single DataFrame
        df_combined = pd.concat(dfs)

        # Debugging: Check if df_combined is empty
        if df_combined.empty:
            print(f"Warning: Combined DataFrame is empty for model '{model}'. Skipping plot.")
            exit()

        # Create and save the plot for the current denoising method
        _, ax = plt.subplots(figsize=(12, 8))
        if df_combined.empty or df_combined.shape[1] == 0:
            print(f"Warning: Empty DataFrame for model '{model}'. Skipping plot.")
            continue
        # Reset index for compatibility with seaborn
        df_combined = df_combined.T.reset_index().rename(columns={'index': 'Models'})

        # Melt the DataFrame for seaborn
        df_melted = df_combined.melt(id_vars='Models', var_name='Scenario', value_name='R2')

        # Create the barplot
        sns.barplot(
            data=df_melted,
            x='Models',
            y='R2',
            hue='Scenario',
            ax=ax,
            palette='ch:start=.5,rot=-0.6,dark=.5,light=.8'
        )

        # Add values above the bars
        for container in ax.containers:
            for bar_var in container:
                height = bar_var.get_height()
                ax.text(
                    bar_var.get_x() + bar_var.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        # Configure plot title and labels
        synthetic_str = 'real' if data_origin == 'real' else 'synthetic'
        dataset_str = dataset.upper()
        data_str = f'{synthetic_str} {dataset_str}'
        ax.set_title(
            f"Denoising method: {model.upper()} - Data: {data_str}\n\
            R2 score per model and scenario",
            fontsize=14
        )
        ax.set_ylabel("R2 score", fontsize=12)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_xticks(range(len(df_combined['Models'])))
        ax.set_xticklabels(df_combined['Models'], rotation=0, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.22), fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the figure for the current denoising method
        fig_path = os.path.join(out_path, data_type, data_origin, dataset)
        make_dir(fig_path)
        fig_path = os.path.join(fig_path, f'{model}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        progress_bar.update(1)


def main():
    """
    Main function to analyze denoising performance across datasets and models.
    """
    selected_folder = DATA_PATH

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

                # Extract metrics data
                is_real = data_origin == 'real'
                sigma = '0.05' if not is_real else ''
                metrics_dict = get_metrics_data(dataset_path, sigma, is_real)
                # Process and plot metrics
                process_and_plot_metrics(metrics_dict, data_type, data_origin, dataset, OUT_PATH)

# Main
if __name__ == "__main__":
    main()
