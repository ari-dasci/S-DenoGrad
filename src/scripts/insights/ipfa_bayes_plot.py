"""
Script for generating Bayesian comparison plots for IPFA (Improved Performance Factor Analysis).

This script analyzes and compares different denoising methods using Bayesian analysis, generating
comparative plots for different data types (tabular/time series) and sources (real/synthetic).
"""

# External libraries
import os
import sys
import json
from itertools import islice
import numpy as np
import baycomp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Global variables and path configuration
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'out')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'insights', 'ipfa', 'bayes_plot')

assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Local imports
from utils import show_menu, make_dir


def generate_comparison_plot(gradient_data, method_data, names, output_path, plot_type):
    """
    Generate and save a Bayesian comparison plot.

    Args:
        gradient_data (np.ndarray): Data for the gradient method
        method_data (np.ndarray): Data for the comparison method
        names (list): Names of the methods being compared
        output_path (str): Path to save the plot
        plot_type (str): Type of plot (dod, ood, or doo)
    """
    probs, fig = baycomp.two_on_multiple(
        gradient_data,
        method_data,
        rope=0.01,
        runs=100000,
        plot=True,
        names=names
    )
    fig.savefig(
        os.path.join(output_path, f'gradient_v_{names[1]}_{plot_type}.png'),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def process_metrics_file(file_path, noise_lvl, min_kl_div, max_kl_div):
    """
    Process a single metrics file and calculate R2 scores.

    Args:
        file_path (str): Path to the metrics file
        noise_lvl (float): Noise level to analyze
        min_kl_div (float): Minimum KL divergence value
        max_kl_div (float): Maximum KL divergence value

    Returns:
        tuple: Three lists containing dod_r2, ood_r2, and doo_r2 scores
    """
    method = file_path.split(os.path.sep)[-1].split('_')[0]
    is_real = 'real' in file_path

    with open(file_path, 'r', encoding='utf-8') as file:
        metrics = json.load(file)

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

    # List of DataFrames to process
    dfs = [orig_df, dod_df, doo_df, ood_df]

    # Get intersection of columns across all DataFrames
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    # Filter DataFrames to keep only common columns
    dfs_filtered = [df[list(common_cols)].loc[['R2']] for df in dfs]

    # Concatenate DataFrames aligning by common columns
    df_combined = pd.concat(dfs_filtered, axis=0)
    df_combined.dropna(axis=1, inplace=True)
    df_combined = df_combined.clip(lower=0.0)

    # Add normalized Kullback-Leibler divergence
    kl_key = 'kl_orig_denoised_' if is_real else 'kl_noisy_denoised_'
    kl_divs = [v for k, v in denoised_dict.items() if kl_key in k]
    
    kl_mean = np.mean(kl_divs)
    kl_mean_norm = (kl_mean - min_kl_div) / (max_kl_div - min_kl_div)

    df_combined.iloc[0] += 1
    df_combined.iloc[1:] += (1 - kl_mean_norm)

    # Add labels for each DataFrame
    new_index = ['Noisy', 'Denoised-Denoised', 'Denoised-Original', 'Original-Denoised']
    df_combined.index = new_index
    df_combined = df_combined.sort_index(axis=1)

    # Extract R2 scores
    dod_current_r2 = [0 if x is None else x for x in df_combined.iloc[1].tolist()]
    doo_current_r2 = [0 if x is None else x for x in df_combined.iloc[2].tolist()]
    ood_current_r2 = [0 if x is None else x for x in df_combined.iloc[3].tolist()]

    return dod_current_r2, ood_current_r2, doo_current_r2


# Main
if __name__ == "__main__":
    noise_lvl = 0.05
    # Folder selection
    selected_folder = show_menu(DATA_PATH)

    while os.path.isfile(selected_folder):
        selected_folder = show_menu(DATA_PATH)
        print(f"\n\t» File: {selected_folder}. It was not a folder.")

    # Initialize dictionaries to store results by data type
    results_by_type = {
        'tabular': {
            'real': {'dod_r2': {}, 'ood_r2': {}, 'doo_r2': {}},
            'synthetic': {'dod_r2': {}, 'ood_r2': {}, 'doo_r2': {}}
        },
        'time_series': {
            'real': {'dod_r2': {}, 'ood_r2': {}, 'doo_r2': {}},
            'synthetic': {'dod_r2': {}, 'ood_r2': {}, 'doo_r2': {}}
        }
    }

    # Process files and accumulate results by type
    for root, dirs, files in os.walk(selected_folder):
        if dirs:
            continue

        list_of_files = [os.path.join(root, file) for file in files if 'metrics' in file]
        if not list_of_files:
            continue

        # Determine if we're in real/synthetic and tabular/time_series folder
        data_type = 'real' if 'real' in root else 'synthetic'
        data_format = 'time_series' if 'time_series' in root else 'tabular'

        # Get the max and min value of Kullback-Leibler divergence
        kl_divs = []
        for file in tqdm(list_of_files, desc="Getting KL-divergence values"):
            with open(file, 'r', encoding='utf-8') as archivo:
                metrics = json.load(archivo)

            is_real = 'real' in file
            kl_key = 'kl_orig_denoised_' if is_real else 'kl_noisy_denoised_'
            denoised_metrics = (metrics['denoised'] if is_real
                              else metrics[f'{noise_lvl}'])

            kl_divs.extend([v for k, v in denoised_metrics.items() if kl_key in k])

        max_kl_div = max(kl_divs)
        min_kl_div = min(kl_divs)

        # Process current folder files
        for file in tqdm(list_of_files, desc=f"Computing IPFA scores for {data_format}/{data_type}"):
            method = file.split(os.path.sep)[-1].split('_')[0]

            # Initialize lists if method doesn't exist
            if method not in results_by_type[data_format][data_type]['dod_r2']:
                results_by_type[data_format][data_type]['dod_r2'][method] = []
                results_by_type[data_format][data_type]['ood_r2'][method] = []
                results_by_type[data_format][data_type]['doo_r2'][method] = []

            # Process metrics file and get R2 scores
            dod_r2, ood_r2, doo_r2 = process_metrics_file(
                file,
                noise_lvl,
                min_kl_div,
                max_kl_div
            )

            # Convert numpy arrays to lists if necessary
            if isinstance(results_by_type[data_format][data_type]['dod_r2'][method], np.ndarray):
                for metric_type in ['dod_r2', 'ood_r2', 'doo_r2']:
                    results_by_type[data_format][data_type][metric_type][method] = (
                        results_by_type[data_format][data_type][metric_type][method].tolist()
                    )

            # Extend results with new values
            results_by_type[data_format][data_type]['dod_r2'][method].extend(dod_r2)
            results_by_type[data_format][data_type]['ood_r2'][method].extend(ood_r2)
            results_by_type[data_format][data_type]['doo_r2'][method].extend(doo_r2)

    # Create plots for each data type
    for data_format, format_results in results_by_type.items():
        for data_type, metrics in format_results.items():
            if not metrics['dod_r2']:  # Skip if no data for this type
                continue

            print(f"\nProcessing {data_format}/{data_type}")

            # Extract and convert gradient data
            dod_gradient = np.array(metrics['dod_r2'].pop('gradient', []))
            ood_gradient = np.array(metrics['ood_r2'].pop('gradient', []))
            doo_gradient = np.array(metrics['doo_r2'].pop('gradient', []))

            # Convert remaining methods to numpy arrays
            for metric_type in ['dod_r2', 'ood_r2', 'doo_r2']:
                metrics[metric_type] = {
                    k: np.array(v) for k, v in sorted(metrics[metric_type].items())
                }

            # Create plots for each method
            for method, dod_values in metrics['dod_r2'].items():
                ood_values = metrics['ood_r2'][method]
                doo_values = metrics['doo_r2'][method]
                names = ['Ours', method]

                # Find maximum length among all arrays
                max_len = max(len(arr) for arr in [
                    dod_gradient, ood_gradient, doo_gradient,
                    dod_values, ood_values, doo_values
                ])

                # Pad arrays with zeros to match maximum length
                arrays_to_pad = {
                    'dod': (dod_gradient, dod_values),
                    'ood': (ood_gradient, ood_values),
                    'doo': (doo_gradient, doo_values)
                }

                padded_arrays = {
                    key: tuple(
                        np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0)
                        for arr in arrays
                    )
                    for key, arrays in arrays_to_pad.items()
                }

                print(f"\nMethod: {method}")
                print(f"Original gradient length: {len(dod_gradient)}")
                print(f"Original method length: {len(dod_values)}")
                print(f"Final padding length: {max_len}")

                # Create and save plots
                fig_path = os.path.join(OUT_PATH, data_format, data_type)
                make_dir(fig_path)

                # Generate all comparison plots
                for plot_type, (grad_pad, values_pad) in padded_arrays.items():
                    generate_comparison_plot(
                        grad_pad,
                        values_pad,
                        names,
                        fig_path,
                        plot_type
                    )
