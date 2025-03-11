# Libs
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from itertools import islice
import seaborn as sns

# Global vars
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'out')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'kl_divergence')

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


# Main
if __name__ == "__main__":
    # Folder or file selection
    selected_folder = show_menu(DATA_PATH)
    list_of_files = []

    # Check that a folder has been selected.
    if os.path.isfile(selected_folder):
        raise ValueError('A folder must be selected.')

    noise_lvl = 0.05
    # Walk through the folder until reach a leaf folder.
    for root, dirs, files in os.walk(selected_folder):
        if dirs:
            continue

        # Stablish if the folder contains real/synthetic and tabular/time_series data.
        is_real = 'real' in root
        is_tabular = 'tabular' in root

        # Go through all the files to make the comparison with gradient metrics.
        kl_orig_denoised_dict = {}
        for file in tqdm(files):
            method = file.split('_')[0]
            if 'metrics' not in file:
                continue

            kl_orig_denoised_dict[method] = {}

            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                file_metrics = json.load(f)

            # Get the denoised_over_denoised and orig_over_denoised metrics from file file.
            denoised_file_metrics = {}
            if not is_real:
                denoised_file_metrics = dict(islice(file_metrics.items(), 1, None))
                denoised_file_metrics = denoised_file_metrics[f'{noise_lvl}']
            else:
                denoised_file_metrics = file_metrics['denoised']

            current_kl_metrics = {
                k.split('kl_orig_denoised_')[1]: v
                for k, v in denoised_file_metrics.items()
                if 'kl_orig_denoised_' in k
            }
            kl_orig_denoised_dict[method].update(current_kl_metrics)

        df_kl = pd.DataFrame(kl_orig_denoised_dict)
        df_kl = df_kl[sorted(df_kl.columns)]

        # Make and save KL-HEATMAP
        # ---------------------------------------------------------------------------------------- #
        fig, ax = plt.subplots(figsize=(10, 6))
        if df_kl.empty or df_kl.shape[1] == 0:
            raise ValueError(f"File: {file}. Empty Dataframe.")

        sns.heatmap(df_kl, cmap="coolwarm", linewidths=0.01)

        path_parts = root.split(os.path.sep)
        dataset = path_parts[-1]
        real_or_synthetic = path_parts[-2]
        tabular_or_ts = path_parts[-3]
        # Guardar la figura
        dataset_str = dataset.upper()
        noise_str = ''
        if not is_real:
            noise_str = f'_s{noise_lvl}'

            if not is_tabular:
                dataset_str = ''
                dataset = 'synthetic'
                real_or_synthetic = path_parts[-1]
                tabular_or_ts = path_parts[-2]

        data_str = f'{tabular_or_ts} {real_or_synthetic} {dataset_str}{noise_str}'
        # Configuración de la gráfica
        ax.set_title(
            f"KL divergence per Variable and Method\nData: {data_str}",
            fontsize=14
        )
        ax.set_ylabel("Methods", fontsize=12)
        ax.set_xlabel("Variables", fontsize=12)
        # ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
        make_dir(fig_path)
        fig_path = os.path.join(fig_path, f'{dataset}_heatmap.png')
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()


        # Make and save KL-BAR PLOT
        # ---------------------------------------------------------------------------------------- #
        fig, ax = plt.subplots(figsize=(10, 6))
        if df_kl.empty or df_kl.shape[1] == 0:
            raise ValueError("El DataFrame está vacío o no tiene columnas para graficar.")

        # df_kl.mean().plot(
        #     kind='bar',
        #     ax=ax,
        #     width=0.7,
        #     color='skyblue'
        # )
        
        sns.barplot(
            data=df_kl.mean(),
            ax=ax,
            palette='viridis'
        )
        # Añadir valores encima de las barras
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

        path_parts = root.split(os.path.sep)
        dataset = path_parts[-1]
        real_or_synthetic = path_parts[-2]
        tabular_or_ts = path_parts[-3]
        # Guardar la figura
        dataset_str = dataset.upper()
        noise_str = ''
        if not is_real:
            noise_str = f'_s{noise_lvl}'

            if not is_tabular:
                dataset_str = ''
                dataset = 'synthetic'
                real_or_synthetic = path_parts[-1]
                tabular_or_ts = path_parts[-2]

        data_str = f'{tabular_or_ts} {real_or_synthetic} {dataset_str}{noise_str}'
        # Configuración de la gráfica
        ax.set_title(
            f"Mean KL divergence per Method\nData: {data_str}",
            fontsize=14
        )
        ax.set_ylabel("Methods", fontsize=12)
        ax.set_xlabel("KL mean", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)


        fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
        make_dir(fig_path)
        fig_path = os.path.join(fig_path, f'{dataset}_barplot.png')
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()