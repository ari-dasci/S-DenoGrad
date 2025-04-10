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
    noise_lvl = 0.05

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
                    gradient_files = [f for f in os.listdir(method_path) if 'metrics' in f]
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
                    gradient_metrics_df['method'] = 'DLNR'

                for method in tqdm(os.listdir(dataset_path)):
                    if method == 'dlnr':
                        continue

                    method_path = os.path.join(dataset_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    # Read the metrics files
                    files = [f for f in os.listdir(method_path) if 'metrics' in f]
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
                        fig_path = os.path.join(OUT_PATH, data_type, data_origin, dataset, train_test)
                        make_dir(fig_path)
                        fig_path = os.path.join(fig_path, f'{method}.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                        plt.close()

                    # # Crear la gráfica de barras
                    # plt.figure(figsize=(16, 10))
                    # sns.barplot(
                    #     data=df_combined,
                    #     x='model',
                    #     y='R2',
                    #     hue='method',
                    #     palette='tab10',
                    #     dodge=True,
                    #     errorbar=None
                    # )

                    # # Añadir etiquetas encima de las barras
                    # for container in plt.gca().containers:
                    #     plt.gca().bar_label(container, fmt='%.2f', fontsize=10, padding=3)

                    # # Añadir etiquetas y título
                    # plt.title('R2 Scores by Model, Method, and Train-Test Configuration',
                    #           fontsize=16)
                    # plt.ylabel('R2 Score', fontsize=12)
                    # plt.xlabel('Model', fontsize=12)
                    # plt.legend(title='Method', fontsize=10, title_fontsize=12)
                    # plt.grid(axis='y', linestyle='--', alpha=0.7)

                    # # Save the figure
                    # fig_path = os.path.join(OUT_PATH, data_type, data_origin, dataset)
                    # make_dir(fig_path)
                    # fig_path = os.path.join(fig_path, f'{method}.png')
                    # plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    # plt.close()
