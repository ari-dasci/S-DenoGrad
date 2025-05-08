"""
This script provides functionality to compare and visualize the original and denoised datasets 
using scatter plots. It supports both real and synthetic datasets and allows for optional 
Gaussian noise addition to the original dataset for visualization purposes.

Functions:
    plot_datasets(original_path, denoised_path, dataset_name, model_name, num_points=None, 
        is_synthetic=False):
            is_synthetic (bool, optional): If True, adds Gaussian noise to the original dataset for comparison.

Command-line Interface:
    This script can be executed from the command line with the following arguments:
        --data_type (str): Type of data ('tabular' or 'time_series').
        --data_origin (str): Origin of data ('real' or 'synthetic').
        --dataset_name (str): Name of the dataset.
        --model_name (str): Name of the denoising model.
        --num_points (int, optional): Number of points to plot (default: all).

Execution:
    The script reads the original and denoised datasets from the specified paths, validates their 
    existence, and generates comparison plots. The output is saved in a structured directory 
    under 'out/insights/dataset_comparison/'.

Example Usage:
    python plot_denoised_vs_original.py --data_type tabular --data_origin real \
        --dataset_name my_dataset --model_name my_model --num_points 100
"""
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_datasets(original_path, denoised_path, dataset_name, model_name, num_points=None,
                  is_synthetic=False):
    """
    Plots a comparison of the original, denoised, and optionally noisy datasets.
    Parameters:
        original_path (str): Path to the original dataset file in Parquet format.
        denoised_path (str): Path to the denoised dataset file in Parquet format.
        dataset_name (str): Name of the dataset, used for output directory naming.
        model_name (str): Name of the model, used for output file naming.
        num_points (int, optional): Number of data points to plot. If None, all points are used.
        is_synthetic (bool, optional): If True, adds Gaussian noise to the original dataset for
            comparison.
    Raises:
        AssertionError: If the columns of the original and denoised datasets do not match.
    Saves:
        A PNG file containing the comparison plots in the directory:
        out/insights/dataset_comparison/<dataset_name>/
            <model_name>_original_vs_denoised_comparison.png
    Notes:
        - The function scales the original dataset using `StandardScaler` before plotting.
        - If `is_synthetic` is True, Gaussian noise with a mean of 0 and standard deviation of 0.05
          is added to the original dataset for visualization.
        - The function ensures consistent plotting by resetting the indices of both datasets.
        - Unused subplots in the grid layout are hidden.
    """
    # Load the original dataset
    original_df = pd.read_parquet(original_path)
    # Load the denoised dataset
    denoised_df = pd.read_parquet(denoised_path)

    # Limit the number of points if specified
    if num_points:
        original_df = original_df.head(num_points)
        denoised_df = denoised_df.head(num_points)

    # Ensure the indices are reset for both datasets
    # This is important for consistent plotting
    original_df = original_df.reset_index(drop=True)
    denoised_df = denoised_df.reset_index(drop=True)

    # Ensure both datasets have the same columns
    assert list(original_df.columns) == list(denoised_df.columns), \
        "Columns do not match between datasets."

    # Scale the original dataset
    scaler = StandardScaler()
    original_df = pd.DataFrame(
        scaler.fit_transform(original_df),
        columns=original_df.columns,
        index=original_df.index
    )

    # Add Gaussian noise if the dataset is synthetic
    noisy_df = None
    if is_synthetic:
        noise = np.random.normal(0, 0.05, original_df.shape)
        noisy_df = original_df + noise

    # Determine the layout of the subplots
    num_vars = len(original_df.columns)
    grid_size = math.ceil(math.sqrt(num_vars))
    _, axes = plt.subplots(
        grid_size, grid_size,
        figsize=(5 * grid_size, 5 * grid_size),
        squeeze=False
    )
    axes = axes.flatten()[:num_vars]  # Only keep the required number of subplots

    # Flatten axes for easier iteration
    axes = axes.flatten()

    for i, column in enumerate(original_df.columns):
        axes[i].scatter(original_df.index, original_df[column], label='Original', alpha=0.7, s=10)
        if is_synthetic:
            axes[i].scatter(noisy_df.index, noisy_df[column], label='Noisy', alpha=0.7, s=10)
        axes[i].scatter(denoised_df.index, denoised_df[column], label='Denoised', alpha=0.7, s=10)
        axes[i].set_title(f'Original vs Noisy vs Denoised: {column}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Values')
        axes[i].legend()
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Use the dataset name and model name for the output directory and file
    output_dir = os.path.join('out', 'insights', 'dataset_comparison', dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot with the model name included in the file name
    output_file = os.path.join(output_dir, f'{model_name}_original_vs_denoised_comparison.png')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Plot original vs denoised dataset.")
    parser.add_argument('--data_type', type=str, required=True, choices=['tabular', 'time_series'],
                        help="Type of data (tabular or time_series).")
    parser.add_argument('--data_origin', type=str, required=True, choices=['real', 'synthetic'],
                        help="Origin of data (real or synthetic).")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset.")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Name of the denoising model.")
    parser.add_argument('--num_points', type=int, default=None,
                        help="Number of points to plot (default: all).")
    args = parser.parse_args()

    # Define paths
    base_path = os.path.join('data', args.data_type, args.data_origin, args.dataset_name)
    original_path = os.path.join(base_path, 'clean.parquet')
    denoised_path = os.path.join(base_path, 'denoised', f"{args.model_name}_denoised.parquet")

    # Check if files exist
    if not os.path.exists(original_path):
        print(f"Original dataset not found: {original_path}")
        sys.exit(1)
    if not os.path.exists(denoised_path):
        print(f"Denoised dataset not found: {denoised_path}")
        sys.exit(1)

    # Plot datasets
    is_synthetic = args.data_origin == 'synthetic'
    plot_datasets(
        original_path,
        denoised_path,
        args.dataset_name,
        args.model_name,
        args.num_points,
        is_synthetic
    )
