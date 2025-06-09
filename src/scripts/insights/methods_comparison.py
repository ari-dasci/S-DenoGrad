"""
This script generates comparative visualizations of R2 scores for different models and methods
based on metrics extracted from JSON files. The script processes data for both tabular and 
time-series datasets, distinguishing between real and synthetic data origins.

Modules:
    - os: Provides functions for interacting with the operating system.
    - sys: Provides access to system-specific parameters and functions.
    - json: Handles JSON file reading and writing.
    - matplotlib.pyplot: Used for creating visualizations.
    - seaborn: Provides a high-level interface for drawing attractive statistical graphics.
    - pandas: Used for data manipulation and analysis.
    - tqdm: Displays progress bars for loops.
    - src.libs.utils.make_dir: Utility function for creating directories.

Global Variables:
    - _CURRENT_DIR: The current working directory.
    - _FOLDERS: List of folder names in the current directory path.
    - _PROJECT_FOLDER_INDEX: Index of the project folder in the directory path.
    - DATA_PATH: Path to the output data directory.
    - OUT_PATH: Path to the output directory for saving visualizations.

Main Functionality:
    - Iterates through data types ('tabular', 'time_series') and data origins ('real', 'synthetic').
    - Reads R2 metrics from JSON files for each dataset and method.
    - Processes metrics to filter and structure data for visualization.
    - Generates bar plots comparing R2 scores across models and methods for different train-test splits.
    - Saves the generated plots to the specified output directory.

Key Steps:
    1. Traverse the folder structure to locate datasets and methods.
    2. Read and process metrics files for each method.
    3. Combine metrics from different methods into a single DataFrame.
    4. Filter and group data based on train-test splits.
    5. Create bar plots for each train-test split, showing R2 scores by model and method.
    6. Save the plots to the appropriate output directory.

Usage:
    Run the script directly to generate and save the visualizations. Ensure that the required
    folder structure and JSON metrics files are present in the specified paths.

Notes:
    - The script assumes a specific folder structure and naming convention for metrics files.
    - The 'dlnr' method is treated as a baseline and processed separately.
    - The script skips datasets or methods if required files or directories are missing.
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
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'insights', 'r2', 'methods_comparison')

# Main
if __name__ == "__main__":
    NOISE_LVL = '0.05'

    # Walk through the folder structure
    for data_type in ['tabular', 'time_series']:
        data_type_path = os.path.join(DATA_PATH, data_type)
        if not os.path.exists(data_type_path):
            continue

        for data_origin in ['real', 'synthetic']:
            origin_path = os.path.join(data_type_path, data_origin)
            if not os.path.exists(origin_path):
                continue

            is_real = data_origin == 'real'
            if is_real:
                NOISE_LVL = ''
            for dataset in os.listdir(origin_path):
                dataset_path = os.path.join(origin_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                gradient_metrics = {}
                for method in os.listdir(dataset_path):
                    if method != 'dlnr':
                        continue

                    method_path = os.path.join(dataset_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    # Read the metrics files
                    gradient_files = [
                        f for f in os.listdir(method_path)
                        if 'metrics' in f and NOISE_LVL in f
                    ]
                    if not gradient_files:
                        continue

                    gradient_file = os.path.join(method_path, gradient_files[0])
                    with open(gradient_file, 'r', encoding='utf-8') as f:
                        gradient_metrics_file = json.load(f)

                    # Initialize gradient_metrics as a nested dictionary
                    gradient_metrics = gradient_metrics_file['XAI']
                    gradient_metrics = {
                        train_test_key: {
                            model_key: metrics.get('R2') if metrics.get('R2') > 0 else 0
                            for model_key, metrics in model_dict.items()
                            if metrics.get('R2') is not None
                        }
                        for noisy_key, train_test_dict in gradient_metrics.items()
                        for train_test_key, model_dict in train_test_dict.items()
                    }

                    # gradient_metrics_df.columns = ['Train-Test', 'Model', 'R2']
                    gradient_metrics_df = pd.DataFrame(gradient_metrics).reset_index().melt(
                        id_vars='index',
                        var_name='train_test',
                        value_name='R2'
                    )
                    gradient_metrics_df['method'] = 'DenoGrad'

                for method in tqdm(os.listdir(dataset_path)):
                    if method == 'dlnr':
                        continue

                    method_path = os.path.join(dataset_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    # Read the metrics files
                    files = [
                        f for f in os.listdir(method_path)
                        if 'metrics' in f and NOISE_LVL in f
                    ]
                    if not files:
                        continue

                    file = os.path.join(method_path, files[0])
                    with open(file, 'r', encoding='utf-8') as f:
                        metrics_file = json.load(f)

                    # Extract the metrics
                    metrics = metrics_file['XAI']
                    metrics = {
                        train_test_key: {
                            model_key: metrics.get('R2') if metrics.get('R2') > 0 else 0
                            for model_key, metrics in model_dict.items()
                            if metrics.get('R2') is not None
                        }
                        for noisy_key, train_test_dict in metrics.items()
                        for train_test_key, model_dict in train_test_dict.items()
                    }

                    metrics_df = pd.DataFrame(metrics).reset_index().melt(
                        id_vars='index',
                        var_name='train_test',
                        value_name='R2'
                    )
                    metrics_df['method'] = method.upper()

                    # Combinar ambos DataFrames
                    df_combined = pd.concat([gradient_metrics_df, metrics_df])
                    df_combined.rename(columns={'index': 'model'}, inplace=True)

                    # Filtrar las entradas que no sean 'train_noisy_test_noisy'
                    df_combined = df_combined[df_combined['train_test'] != 'train_noisy_test_noisy']

                    # Separar en diferentes DataFrames según el valor de 'train_test'
                    train_test_groups = df_combined.groupby('train_test')

                    # Crear una gráfica de barras para cada grupo
                    for train_test, group_df in train_test_groups:
                        plt.figure(figsize=(16, 10))
                        sns.barplot(
                            data=group_df,
                            x='model',
                            y='R2',
                            hue='method',
                            palette='tab10',
                            dodge=True,
                            errorbar=None
                        )

                        # Añadir etiquetas encima de las barras
                        for container in plt.gca().containers:
                            plt.gca().bar_label(container, fmt='%.2f', fontsize=10, padding=3)

                        # Añadir etiquetas y título
                        plt.title(f'R2 Scores by Model and Method ({train_test})', fontsize=16)
                        plt.ylabel('R2 Score', fontsize=12)
                        plt.xlabel('Model', fontsize=12)
                        plt.legend(title='Method', fontsize=10, title_fontsize=12)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)

                        # Guardar la figura
                        fig_path = os.path.join(OUT_PATH, data_type, data_origin, dataset,
                                                train_test)
                        make_dir(fig_path)
                        fig_path = os.path.join(fig_path, f'{method}.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                        plt.close()
