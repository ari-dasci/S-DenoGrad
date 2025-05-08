# Libs
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import baycomp
import numpy as np
# Local imports
sys.path.append(os.getcwd())
from src.libs.utils import make_dir

# Global vars
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'out')
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'insights', 'bayes')

# Main
if __name__ == "__main__":
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
            NOISE_LVL = '' if is_real else '0.05'

            ood_r2 = {} # train_original_test_denoised
            nod_r2 = {} # train_noisy_test_denoised
            doo_r2 = {} # train_denoised_test_original
            don_r2 = {} # train_denoised_test_noisy
            dod_r2 = {} # train_denoised_test_denoised
            for dataset in os.listdir(origin_path):
                dataset_path = os.path.join(origin_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                # Iterate through the methods and read the metrics files
                for method in os.listdir(dataset_path):
                    method_path = os.path.join(dataset_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    for file in os.listdir(method_path):
                        if 'metrics' not in file or NOISE_LVL not in file:
                            continue

                        # Read metrics
                        metrics_file = os.path.join(method_path, file)
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            current_metrics = json.load(f)

                        original_current_metrics = {}
                        current_metrics = current_metrics.get('XAI', {})
                        if not is_real:
                            original_current_metrics = current_metrics.get(
                                'orig', {})
                        noisy_current_metrics = current_metrics.get(
                            'noisy', {})
                        denoised_current_metrics = current_metrics.get(
                            'denoised', {})

                        ood_current_metrics = {}
                        doo_current_metrics = {}
                        if not is_real:
                            ood_current_metrics = original_current_metrics.get(
                                'train_original_test_denoised', {})
                            doo_current_metrics = denoised_current_metrics.get(
                                'train_denoised_test_original', {})
                        nod_current_metrics = noisy_current_metrics.get(
                            'train_noisy_test_denoised', {})
                        don_current_metrics = denoised_current_metrics.get(
                            'train_denoised_test_noisy', {})
                        dod_current_metrics = denoised_current_metrics.get(
                            'train_denoised_test_denoised', {})

                        ood_current_r2 = []
                        doo_current_r2 = []
                        if data_type == 'tabular':
                            if not is_real:
                                ood_current_r2 = [
                                    metrics.get('R2', 0)
                                    for key, metrics in ood_current_metrics.items()
                                    if key not in ['arima', 'auto_arima']
                                ]
                                doo_current_r2 = [
                                    metrics.get('R2', 0)
                                    for key, metrics in doo_current_metrics.items()
                                    if key not in ['arima', 'auto_arima']
                                ]
                            nod_current_r2 = [
                                metrics.get('R2', 0) for key, metrics in nod_current_metrics.items()
                                if key not in ['arima', 'auto_arima']
                            ]
                            don_current_r2 = [
                                metrics.get('R2', 0) for key, metrics in don_current_metrics.items()
                                if key not in ['arima', 'auto_arima']
                            ]
                            dod_current_r2 = [
                                metrics.get('R2', 0) for key, metrics in dod_current_metrics.items()
                                if key not in ['arima', 'auto_arima']
                            ]
                        else:
                            if not is_real:
                                ood_current_r2 = [
                                    metrics.get('R2', 0)
                                    for _, metrics in ood_current_metrics.items()
                                ]
                                doo_current_r2 = [
                                    metrics.get('R2', 0)
                                    for _, metrics in doo_current_metrics.items()
                                ]
                            nod_current_r2 = [
                                metrics.get('R2', 0) for _, metrics in nod_current_metrics.items()
                            ]
                            don_current_r2 = [
                                metrics.get('R2', 0) for _, metrics in don_current_metrics.items()
                            ]
                            dod_current_r2 = [
                                metrics.get('R2', 0) for _, metrics in dod_current_metrics.items()
                            ]


                        ood_current_r2 = [0 if x is None else x for x in ood_current_r2]
                        nod_current_r2 = [0 if x is None else x for x in nod_current_r2]
                        doo_current_r2 = [0 if x is None else x for x in doo_current_r2]
                        don_current_r2 = [0 if x is None else x for x in don_current_r2]
                        dod_current_r2 = [0 if x is None else x for x in dod_current_r2]

                        # Check not all values are 0.
                        if not is_real:
                            # print(metrics_file)
                            assert any(ood_current_r2)
                            assert any(doo_current_r2)
                        assert any(nod_current_r2)
                        assert any(don_current_r2)
                        assert any(dod_current_r2)

                        # Extend the lists with the new values.
                        method_name = file.split('_')[0]
                        ood_r2.setdefault(method_name, []).extend(ood_current_r2)
                        nod_r2.setdefault(method_name, []).extend(nod_current_r2)
                        doo_r2.setdefault(method_name, []).extend(doo_current_r2)
                        don_r2.setdefault(method_name, []).extend(don_current_r2)
                        dod_r2.setdefault(method_name, []).extend(dod_current_r2)

            # Generate and save plots
            dict_names = ['ood', 'nod', 'doo', 'don', 'dod']
            dicts = [ood_r2, nod_r2, doo_r2, don_r2, dod_r2]
            for dict_name, r2_metrics in zip(dict_names, dicts):
                for model_name, model_r2 in r2_metrics.items():
                    if model_name == 'dlnr':
                        continue

                    gradient_r2 = np.array(r2_metrics['dlnr'])
                    model_r2 = np.array(model_r2)

                    names = ['DLNR (Ours)', model_name]

                    probs, fig = baycomp.two_on_multiple(
                        gradient_r2,
                        model_r2,
                        rope=0.01,
                        runs=100000,
                        plot=True,
                        names=names
                    )

                    path_parts = dataset_path.split(os.path.sep)
                    real_or_synthetic = path_parts[-2]
                    tabular_or_ts = path_parts[-3]

                    # Save the figure in a folder based on the pair and dictionary
                    fig_path = os.path.join(OUT_PATH, tabular_or_ts, real_or_synthetic,
                                            f'DLNR_vs_{model_name}')
                    make_dir(fig_path)
                    fig_path = os.path.join(fig_path, f'{dict_name}.png')
                    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()
