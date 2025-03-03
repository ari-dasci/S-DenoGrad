# Libs
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import islice
import baycomp
import numpy as np

# Global vars
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'out')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'bayes')

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
    for root, _, files in os.walk(init_folder):
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

        # Read gradient metrics to compare it with the other methods.
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

        dod_gradient_metrics = denoised_gradient_metrics['denoised_over_denoised']
        ood_gradient_metrics = denoised_gradient_metrics[ood_tag]

        if is_tabular:
            dod_gradient_r2 = [metrics['R2'] for key, metrics in dod_gradient_metrics.items() if key not in ['arima', 'auto_arima']]
            ood_gradient_r2 = [metrics['R2'] for key, metrics in ood_gradient_metrics.items() if key not in ['arima', 'auto_arima']]
        else:
            dod_gradient_r2 = [metrics['R2'] for _, metrics in dod_gradient_metrics.items()]
            ood_gradient_r2 = [metrics['R2'] for _, metrics in ood_gradient_metrics.items()]
        dod_gradient_r2 = [0 if x is None else x for x in dod_gradient_r2]
        ood_gradient_r2 = [0 if x is None else x for x in ood_gradient_r2]
        # dod_gradient_r2 = dod_gradient_r2[dod_gradient_r2 != None]
        # ood_gradient_r2 = ood_gradient_r2[ood_gradient_r2 != None]
        dod_gradient_r2 = np.array(dod_gradient_r2)
        ood_gradient_r2 = np.array(ood_gradient_r2)

        # Go through all the files to make the comparison with gradient metrics.
        correlation_dict = {}
        last_folder = root.split('/')[-1]
        pbar = tqdm(files, desc='')
        for file in pbar:
            # print(root, file)
            method = file.split('_')[0]
            pbar.set_description(last_folder+'/'+method)
            if 'metrics' not in file or 'gradient' in file:
                continue

            correlation_dict[method] = {}

            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                file_metrics = json.load(f)

            # Get the denoised_over_denoised and orig_over_denoised metrics from file file.
            denoised_file_metrics = {}
            orig_noise_corr = None
            orig_denoised_corr = None
            ood_tag = 'orig_over_denoised'
            if not is_real:
                denoised_file_metrics = dict(islice(file_metrics.items(), 1, None))
                denoised_file_metrics = denoised_file_metrics[f'{noise_lvl}']
                ood_tag = 'noisy_over_denoised'
            else:
                denoised_file_metrics = file_metrics['denoised']

            dod_metrics = denoised_file_metrics['denoised_over_denoised']
            ood_metrics = denoised_file_metrics[ood_tag]

            if is_tabular:
                dod_r2 = [metrics['R2'] for key, metrics in dod_metrics.items() if key not in ['arima', 'auto_arima']]
                ood_r2 = [metrics['R2'] for key, metrics in ood_metrics.items() if key not in ['arima', 'auto_arima']]
            else:
                dod_r2 = [metrics['R2'] for key, metrics in dod_metrics.items()]
                ood_r2 = [metrics['R2'] for key, metrics in ood_metrics.items()]

            # dod_r2 = dod_r2[dod_r2 != None]
            # ood_r2 = ood_r2[ood_r2 != None]
            ood_r2 = [0 if x is None else x for x in ood_r2]
            dod_r2 = [0 if x is None else x for x in dod_r2]
            ood_r2 = np.array(ood_r2)
            dod_r2 = np.array(dod_r2)


            # Crear y guardar la gráfica
            # ------------------------------------------------------------------------------------ #
            # fig, ax = plt.subplots(figsize=(10, 6))
            names = ['Ours', method]
            bar_width = 0.7
            if len(dod_r2) == 0:
                raise ValueError(f"File: {file}. Empty DoD list.")

            probs, fig = baycomp.two_on_multiple(dod_gradient_r2, dod_r2, rope=0.01, plot=True, names=names)

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
            # ax.set_title(
            #     f"Signed Rank Bayesian Test\nDenoised over Denoised\nOurs VS {method} - Data: {data_str}",
            #     fontsize=14
            # )

            fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
            make_dir(fig_path)
            fig_path = os.path.join(fig_path, f'{method}_dod.png')
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()


            # Crear y guardar la gráfica
            # ------------------------------------------------------------------------------------ #
            # fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.7
            if len(ood_r2) == 0:
                raise ValueError(f"File: {file}. Empty OoD list.")

            probs, fig = baycomp.two_on_multiple(ood_gradient_r2, ood_r2, rope=0.01, plot=True, names=names)

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
            # ax.set_title(
            #     f"Signed Rank Bayesian Test\nDenoised over Denoised\nOurs VS {method} - Data: {data_str}",
            #     fontsize=14
            # )

            fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
            make_dir(fig_path)
            fig_path = os.path.join(fig_path, f'{method}_ood.png')
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
