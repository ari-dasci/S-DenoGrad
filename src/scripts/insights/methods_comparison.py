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
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'r2', 'methods_comparison')

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
        raise('A folder must be selected.')

    noise_lvl = 0.05
    # Walk through the folder until reach a leaf folder.
    for root, dirs, files in os.walk(selected_folder):
        if dirs: continue

        # Stablish if the folder contains real/synthetic and tabular/time_series data.
        is_real = 'real' in root
        is_tabular = 'tabular' in root

        # Read the gradient metrics to compare it with the others.
        gradient_i = [f for f in files if 'gradient' in f and 'metrics' in f][0]
        gradient_file = os.path.join(root, gradient_i)
        with open(gradient_file, 'r') as f:
            gradient_metrics = json.load(f)

        # Get the denoised_over_denoised and orig_over_denoised metrics from gradient file.
        ood_tag = 'orig_over_denoised'
        denoised_gradient_metrics = {}
        if not is_real:
            ood_tag = 'noisy_over_denoised'
            denoised_gradient_metrics = dict(islice(gradient_metrics.items(), 1, None))
            denoised_gradient_metrics = denoised_gradient_metrics[f'{noise_lvl}']
        else:
            denoised_gradient_metrics = gradient_metrics['denoised']
    
        dod_gradient_df = pd.DataFrame(denoised_gradient_metrics['denoised_over_denoised'])
        ood_gradient_df = pd.DataFrame(denoised_gradient_metrics[ood_tag])

        # Go through all the files to make the comparison with gradient metrics.
        for file in tqdm(files):
            method = file.split('_')[0]
            if 'gradient' in method or 'metrics' not in file: continue

            with open(os.path.join(root,file), 'r') as f:
                file_metrics = json.load(f)

            # Get the denoised_over_denoised and orig_over_denoised metrics from file file.
            ood_tag = 'orig_over_denoised'
            denoised_file_metrics = {}
            if not is_real:
                ood_tag = 'noisy_over_denoised'
                denoised_file_metrics = dict(islice(file_metrics.items(), 1, None))
                denoised_file_metrics = denoised_file_metrics[f'{noise_lvl}']
            else:
                denoised_file_metrics = file_metrics['denoised']
            
            dod_file_df = pd.DataFrame(denoised_file_metrics['denoised_over_denoised'])
            ood_file_df = pd.DataFrame(denoised_file_metrics[ood_tag])

            # Lista de DataFrames
            dfs = [dod_gradient_df, dod_file_df, ood_gradient_df, ood_file_df]

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
            noise_tag = 'Original'
            if not is_real:
                noise_tag = 'Noisy'

            new_index = [
                'Ours: Denoised-Denoised',
                f'{method.upper()}: Denoised-Denoised',
                f'Ours: {noise_tag}-Denoised',
                f'{method.upper()}: {noise_tag}-Denoised'
            ]
            df_combined.index = new_index
            df_combined = df_combined.sort_index(axis=1)

            # Crear y guardar la gráfica
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.7
            if df_combined.empty or df_combined.shape[1] == 0:
                print(file)
                raise ValueError("El DataFrame está vacío o no tiene columnas para graficar.")

            # Obtener la lista de colores de tab20c en orden secuencial
            tab20c_colors = plt.get_cmap("tab20").colors  # Lista con 20 colores

            # Asignar colores manualmente en orden a cada serie de df_combined
            custom_colors = tab20c_colors[:df_combined.shape[1]]  # Tantos colores como columnas tenga df_combined

            df_combined.T.plot(
                kind='bar',
                ax=ax,
                width=bar_width,
                color=custom_colors
            )

            # Añadir valores encima de las barras
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

            path_parts = os.path.join(root, file).split(os.path.sep)
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
                    dataset = denoising_method
                    real_or_synthetic = path_parts[-2]
                    tabular_or_ts = path_parts[-3]

            data_str = f'{synthetic_str} {dataset_str}{noise_str}'
            # Configuración de la gráfica
            ax.set_title(
                f"Gradient VS {method.upper()} - Data: {data_str}\nR2 score comparison",
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

