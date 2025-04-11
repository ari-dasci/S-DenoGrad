"""
Script to generate bar plots showing the fit coefficient and mutual information 
between original/noisy data and denoised data.

This script processes previously generated metrics files and creates visualizations 
that help understand:
- The quality of fit between original and processed data
- The amount of shared information between different versions of the data  
- Performance differences for real vs synthetic data

The plots are saved in the 'out/insights/ipfa/bar_plots/' directory.
"""


# Libs
import os
import sys
import json
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Global vars
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'out')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'ipfa', 'bar_plots')

assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Local imports
from utils import show_menu, make_dir


# Main
if __name__ == "__main__":
    noise_lvl = 0.05
    # Folder or file selection
    selected_folder = show_menu(DATA_PATH)

    while os.path.isfile(selected_folder):
        selected_folder = show_menu(DATA_PATH)
        print(f"\n\t» File: {selected_folder}. It was not a folder.")

    for root, dirs, files in os.walk(selected_folder):
        if dirs:
            continue

        list_of_files = []
        for file in files:
            if 'metrics' in file:
                list_of_files.append(os.path.join(root, file))

        if not list_of_files:
            print('There are no files to proccess.')
            exit()

        # Get the max and min value of Kullback-Leibler divergence
        kl_divs = []
        for file in tqdm(list_of_files, desc="Getting KL-divergence values"):
            with open(file, 'r', encoding='utf-8') as archivo:
                metrics = json.load(archivo)

            # Stablish if the folder contains real/synthetic and tabular/time_series data.
            is_real = 'real' in file

            # Get the denoised metrics dictionary
            kl_key = 'kl_orig_denoised_'
            denoised_file_metrics = {}
            if not is_real:
                kl_key = 'kl_noisy_denoised_'
                denoised_file_metrics = dict(islice(metrics.items(), 1, None))
                denoised_file_metrics = denoised_file_metrics[f'{noise_lvl}']
            else:
                denoised_file_metrics = metrics['denoised']

            # Add Kullback-Leibler values to the list
            kl_divs.extend([
                v
                for k, v in denoised_file_metrics.items()
                if kl_key in k
            ])

        # Compute the max and min value.
        max_kl_div = max(kl_divs)
        min_kl_div = min(kl_divs)


        # Iter through the files
        for file in tqdm(list_of_files, desc="Creating plots"):
            is_real = 'real' in file
            metrics = None
            with open(file, 'r', encoding='utf-8') as archivo:
                metrics = json.load(archivo)

            ood_tag = 'orig_over_denoised'
            if not is_real:
                denoised_dict = metrics[f'{noise_lvl}']
                orig_df = pd.DataFrame(dict(islice(denoised_dict.items(), 8)))
                ood_tag = 'noisy_over_denoised'
            else:
                orig_df = pd.DataFrame(metrics['orig'])

            denoised_dict = metrics['denoised'] if is_real else denoised_dict
            dod_df = pd.DataFrame(denoised_dict['denoised_over_denoised'])
            doo_df = pd.DataFrame(denoised_dict['denoised_over_orig'])
            ood_df = pd.DataFrame(denoised_dict[ood_tag])

            # Lista de DataFrames
            dfs = [
                orig_df,
                dod_df,
                doo_df,
                ood_df
            ]

            # Obtener la intersección de las columnas en todos los DataFrames
            common_cols = set(dfs[0].columns)
            for df in dfs[1:]:
                common_cols &= set(df.columns)  # Intersección de columnas

            # Filtrar DataFrames para mantener solo las columnas en común
            dfs_filtered = [df[list(common_cols)].loc[['R2']] for df in dfs]

            # Concatenar los DataFrames alineando por columnas comunes
            df_combined = pd.concat(dfs_filtered, axis=0)
            df_combined.dropna(axis=1, inplace=True)
            df_combined = df_combined.clip(lower=0.0)

            # Add the normalized Kullback-Leibler divergence to the dataframe
            kl_key = 'kl_orig_denoised_'
            if not is_real:
                kl_key = 'kl_noisy_denoised_'

            kl_divs = [
                v
                for k, v in denoised_dict.items()
                if kl_key in k
            ]

            kl_mean = np.mean(kl_divs)
            kl_mean_norm = (kl_mean - min_kl_div) / (max_kl_div - min_kl_div)

            df_combined.iloc[0] += 1
            df_combined.iloc[1:] += (1 - kl_mean_norm)

            # Agregamos etiquetas para cada DataFrame
            new_index = ['Noisy', 'Denoised-Denoised', 'Denoised-Original', 'Original-Denoised']
            df_combined.index = new_index
            df_combined = df_combined.sort_index(axis=1)

            # Crear y guardar la gráfica
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.7
            if df_combined.empty or df_combined.shape[1] == 0:
                raise ValueError(f"File: {file}. Empty Dataframe.")
            df_combined.T.plot(kind='bar', ax=ax, width=bar_width, colormap="viridis")
            ax.set_ylim(0, 2)

            # Añadir valores encima de las barras
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

            path_parts = file.split(os.path.sep)
            denoising_method = path_parts[-1].split('_')[0]
            dataset = path_parts[-2]
            real_or_synthetic = path_parts[-3]
            tabular_or_ts = path_parts[-4]
            # Guardar la figura
            synthetic_str = 'real'
            dataset_str = dataset.upper()
            noise_str = ''
            if not is_real:
                synthetic_str = 'synthetic'

                noise_str = f'_s{noise_lvl}'

                if 'time_series' in file:
                    dataset_str = ''
                    dataset = ''
                    real_or_synthetic = path_parts[-2]
                    tabular_or_ts = path_parts[-3]

            data_str = f'{synthetic_str} {dataset_str}{noise_str}'
            # Configuración de la gráfica
            ax.set_title(
                f"Denoising method: {denoising_method.upper()} - Data: {data_str}\nIPFA score per model",
                fontsize=14
            )
            ax.set_ylabel("IPFA score", fontsize=12)
            ax.set_xlabel("Models", fontsize=12)
            ax.set_xticklabels(df_combined.columns, rotation=0, fontsize=11)
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.22), fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            denoising_method += f'{noise_str}.png'

            fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic, dataset)
            make_dir(fig_path)
            fig_path = os.path.join(fig_path, denoising_method)
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
