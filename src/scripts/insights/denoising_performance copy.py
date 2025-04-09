# Libs
import os
import sys
import platform
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from itertools import islice

# Global vars
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'out')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'r2', 'denoising_performance')

assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Local imports
from utils import show_menu, make_dir


def list_files(init_folder):
    """
    Recursively lists all files in the given directory and its subdirectories with relative paths.

    Parameters:
    init_folder (str): The initial directory path.

    Returns:
    list: A list of relative file paths.
    """
    archivos = []
    for root, dirs, files in os.walk(init_folder):
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
        if is_real:
            metrics_dict[model] = file_metrics.get('XAI', {}).get('orig', {})
        else:
            metrics_dict[model] = file_metrics.get('XAI', {}).get('noisy', {})

    return metrics_dict


def process_and_plot_metrics(metrics_dict, data_type, data_origin, dataset, out_path):
    """
    Processes metrics data and generates plots.

    Parameters:
    metrics_dict (dict): Dictionary containing metrics data for each model.
    data_type (str): Type of data (e.g., 'tabular', 'time_series').
    data_origin (str): Origin of data (e.g., 'real', 'synthetic').
    dataset (str): Dataset name.
    out_path (str): Path to save the output plots.
    """
    progress_bar = tqdm(total=len(metrics_dict), desc="Creating plots")

    for model, metrics in metrics_dict.items():
        # Extract metrics for different scenarios
        train_test_scenarios = metrics.keys()
        dfs = []

        for scenario in train_test_scenarios:
            scenario_metrics = metrics[scenario]
            scenario_df = pd.DataFrame.from_dict(scenario_metrics, orient='index')
            print(data_origin, dataset)
            scenario_df = scenario_df.loc[['R2']]  # Only keep R2 scores
            scenario_df.columns = [f"{scenario}_{col}" for col in scenario_df.columns]
            dfs.append(scenario_df)

        # Combine all scenarios into a single DataFrame
        if not dfs:
            continue
        df_combined = pd.concat(dfs, axis=1).dropna(axis=1).clip(lower=0.0)

        # Create and save the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.7
        if df_combined.empty or df_combined.shape[1] == 0:
            raise ValueError(f"Model: {model}. Empty DataFrame.")
        df_combined.T.plot(kind='bar', ax=ax, width=bar_width, colormap="viridis")

        # Add values above the bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

        # Configure plot title and labels
        synthetic_str = 'real' if data_origin == 'real' else 'synthetic'
        dataset_str = dataset.upper()
        data_str = f'{synthetic_str} {dataset_str}'
        ax.set_title(
            f"Denoising method: {model.upper()} - Data: {data_str}\nR2 score per model",
            fontsize=14
        )
        ax.set_ylabel("R2 score", fontsize=12)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_xticklabels(df_combined.columns, rotation=0, fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.22), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the figure
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
