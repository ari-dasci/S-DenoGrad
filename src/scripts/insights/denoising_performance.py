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


# Main
if __name__ == "__main__":
    # Folder or file selection
    selected_file_folder = show_menu(DATA_PATH)
    list_of_files = []

    if os.path.isfile(selected_file_folder):
        list_of_files.append(selected_file_folder)

    else:
        files = list_files(selected_file_folder)
        for f in files:
            if f.endswith('metrics.json'):
                list_of_files.append(f)

    if not list_of_files:
        print('There are no files to proccess.')
        exit()

    # Count the number of plots that are going to be created
    n_files = 0
    for file in list_of_files:
        if 'synthetic' in file:
            coef = 16
            if 'time_series' in file:
                coef = 15
            n_files += coef
        else:
            n_files += 1
    # n_files = len(list_of_files) - synthetic_files + synthetic_files

    # Iter through the files
    progress_bar = tqdm(total=n_files, desc="Creating plots")
    for file in list_of_files:
        is_real = 'real' in file
        metrics = None
        with open(file, 'r') as archivo:
            metrics = json.load(archivo)

        noise_lvls = []
        noise_dfs = [metrics]
        no_noise_df = None
        orig_tag = 'orig'
        ood_tag = 'orig_over_denoised'
        if not is_real:
            synthetic_dict = dict(islice(metrics.items(), 1, None))
            noise_lvls = synthetic_dict.keys()
            noise_dfs = synthetic_dict.values()
            orig_tag = 'no_noise'
            ood_tag = 'noisy_over_denoised'
            no_noise_df = pd.DataFrame(metrics[orig_tag])

        for noise_i, current_df in enumerate(noise_dfs):
            if is_real:
                orig_df = pd.DataFrame(current_df[orig_tag])
            else:
                orig_df = pd.DataFrame(dict(islice(current_df.items(), len(current_df) - 3)))
            denoised_dict = metrics['denoised'] if is_real else current_df
            dod_df = pd.DataFrame(denoised_dict['denoised_over_denoised'])
            doo_df = pd.DataFrame(denoised_dict['denoised_over_orig'])
            ood_df = pd.DataFrame(denoised_dict[ood_tag])

            # Lista de DataFrames
            dfs = [orig_df, dod_df, doo_df, ood_df]
            if not is_real:
                dfs.insert(0, no_noise_df)

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

            # Agregamos etiquetas para cada DataFrame
            new_index = ['Original', 'Denoised-Denoised', 'Denoised-Original', 'Original-Denoised']
            if not is_real:
                new_index.insert(1, 'Noisy')
            df_combined.index = new_index
            df_combined = df_combined.sort_index(axis=1)

            # Crear y guardar la gráfica
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.7
            if df_combined.empty or df_combined.shape[1] == 0:
                raise ValueError(f"File: {file}. Empty Dataframe.")
            df_combined.T.plot(kind='bar', ax=ax, width=bar_width, colormap="viridis")

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

                noise_lvl = (noise_i+1) * 0.01
                noise_str = f'_s{noise_lvl}'
                if noise_lvl == 0.16:
                    noise_str = '_mix'

                if 'time_series' in file:
                    dataset_str = ''
                    dataset = denoising_method
                    real_or_synthetic = path_parts[-2]
                    tabular_or_ts = path_parts[-3]

            data_str = f'{synthetic_str} {dataset_str}{noise_str}'
            # Configuración de la gráfica
            ax.set_title(
                f"Denoising method: {denoising_method.upper()} - Data: {data_str}\nR2 score per model",
                fontsize=14
            )
            ax.set_ylabel("R2 score", fontsize=12)
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

            progress_bar.update(1)
