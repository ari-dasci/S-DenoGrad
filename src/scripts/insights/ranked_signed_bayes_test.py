# Libs
import os
import sys
import json
from itertools import islice
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    archives = []
    for _root, _, _files in os.walk(init_folder):
        for _file in _files:
            absolute_path = os.path.join(_root, _file)
            archives.append(absolute_path)
    return archives


# Main
if __name__ == "__main__":
    # Folder or file selection
    selected_folder = show_menu(DATA_PATH)
    list_of_files = []

    # Check that a folder has been selected.
    if os.path.isfile(selected_folder):
        raise ValueError('A folder must be selected.')

    noise_lvl = 0.05
    # Walk through the folder
    for root, dirs, files in os.walk(selected_folder):
        if not root.endswith(('real', 'synthetic')):
            continue

        # Stablish if the folder contains real/synthetic and tabular/time_series data.
        is_real = 'real' in root
        is_tabular = 'tabular' in root

        dod_r2 = {}
        ood_r2 = {}
        for inner_root, inner_dirs, inner_files in os.walk(root):
            # if there are no files continue exploring
            if not inner_files:
                continue

            for file in inner_files:
                # If the file is not a metrics file: continue exploring
                if not 'metrics' in file:
                    continue

                method = file.split('_')[0]
                if method not in dod_r2:
                    dod_r2[method] = []
                if method not in ood_r2:
                    ood_r2[method] = []

                # Read metrics.
                metrics_file = os.path.join(inner_root, file)
                with open(metrics_file, 'r') as f:
                    current_metrics = json.load(f)

                # Get the denoised_over_denoised and orig_over_denoised metrics from the file.
                ood_tag = 'orig_over_denoised'
                denoised_current_metrics = {}
                if not is_real:
                    ood_tag = 'noisy_over_denoised'
                    denoised_current_metrics = dict(islice(current_metrics.items(), 1, None))
                    denoised_current_metrics = denoised_current_metrics[f'{noise_lvl}']
                else:
                    denoised_current_metrics = current_metrics['denoised']

                dod_current_metrics = denoised_current_metrics['denoised_over_denoised']
                ood_current_metrics = denoised_current_metrics[ood_tag]

                if is_tabular:
                    dod_current_r2 = [
                        metrics['R2'] for key, metrics in dod_current_metrics.items()
                        if key not in ['arima', 'auto_arima']
                    ]
                    ood_current_r2 = [
                        metrics['R2'] for key, metrics in ood_current_metrics.items()
                        if key not in ['arima', 'auto_arima']
                    ]
                else:
                    dod_current_r2 = [metrics['R2'] for _, metrics in dod_current_metrics.items()]
                    ood_current_r2 = [metrics['R2'] for _, metrics in ood_current_metrics.items()]

                dod_current_r2 = [0 if x is None else x for x in dod_current_r2]
                ood_current_r2 = [0 if x is None else x for x in ood_current_r2]

                # Check not all values are 0.
                assert any(dod_current_r2)
                assert any(ood_current_r2)

                # Extend the lists with the new values.
                dod_r2[method] += dod_current_r2
                ood_r2[method] += ood_current_r2


        # Crear y guardar la gráfica
        # ------------------------------------------------------------------------------------ #
        dod_gradient_r2 = np.array(dod_r2['gradient'])
        ood_gradient_r2 = np.array(ood_r2['gradient'])

        dod_r2.pop('gradient')
        ood_r2.pop('gradient')

        dod_r2 = {k: np.array(dod_r2[k]) for k in sorted(dod_r2)}
        ood_r2 = {k: np.array(ood_r2[k]) for k in sorted(ood_r2)}

        for (k1, v1), (k2, v2) in tqdm(zip(dod_r2.items(), ood_r2.items())):
            assert k1 == k2
            names = ['Ours', k1]
            bar_width = 0.7

            probs, fig = baycomp.two_on_multiple(
                dod_gradient_r2,
                v1,
                rope=0.01,
                runs=100000,
                plot=True,
                names=names
            )

            path_parts = root.split(os.path.sep)
            real_or_synthetic = path_parts[-1]
            tabular_or_ts = path_parts[-2]
            # Guardar la figura
            noise_str = ''
            if not is_real:
                noise_str = f'_s{noise_lvl}'

            data_str = f'{tabular_or_ts} {real_or_synthetic}{noise_str}'

            fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
            make_dir(fig_path)
            fig_path = os.path.join(fig_path, f'gradient_v_{k1}_dod.png')
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()


            # Crear y guardar la gráfica
            # ------------------------------------------------------------------------------------ #
            probs, fig = baycomp.two_on_multiple(
                ood_gradient_r2,
                v2,
                rope=0.01,
                runs=100000,
                plot=True,
                names=names
            )

            path_parts = root.split(os.path.sep)
            real_or_synthetic = path_parts[-1]
            tabular_or_ts = path_parts[-2]
            # Guardar la figura
            noise_str = ''
            if not is_real:
                noise_str = f'_s{noise_lvl}'

            data_str = f'{tabular_or_ts} {real_or_synthetic}{noise_str}'

            fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic)
            make_dir(fig_path)
            fig_path = os.path.join(fig_path, f'gradient_v_{k1}_ood.png')
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
