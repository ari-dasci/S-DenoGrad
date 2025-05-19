# pylint: disable=wrong-import-position
"""
title: real_house_prices_exp
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a real experiment with a house_prices dataset.
The experiment consists of generating a house_prices dataset with a polinomial function and
adding Gaussian noise to it. Then, a neural network model is trained to predict the target
variable. Finally, the gradients are used to reduce the noise in the data.
"""

# Libraries #
# ------------------------------------------------------------------------------------------------ #
# Public libraries
import os
import sys
import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
# Denoising libraries
from sklearn.decomposition import PCA
from PyEMD import EMD
from filterpy.kalman import KalmanFilter
import pywt
# Local libraries
sys.path.append(os.getcwd())
from src.libs.models import Trainer, DenoisingAutoencoder, DenseResNetDenoising, GridFullyDenseNN
from src.libs.utils import symmetric_mean_absolute_percentage_error
from src.libs.dataset import TensorDataset, SlidingWindowDataset
from src.libs.dlnr import DLNoiseReduction
from src.libs.experiment_templates import TSExperiment


# Seeds #
# ------------------------------------------------------------------------------------------------ #
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Arguments #
# ------------------------------------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Tabular Experiment")
parser.add_argument(
    '--data_folder', 
    type=str,
    default='daily_climate',
    help="Folder containing the dataset. Default is 'daily_climate'."
)
parser.add_argument(
    '--data_file',
    type=str,
    default='clean.parquet',
    help="Name of the dataset file. Default is 'clean.parquet'"
)
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    help="Device to use for training. Default is 'cuda'."
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help="If set, the script will print the progress of the training and testing."
)
parser.add_argument(
    '--load_original_xai',
    action='store_false',
    help="If set, the script will load the original XAI models checkpoints. Default is False."
)
parser.add_argument(
    '--load_noisy_xai',
    action='store_false',
    help="If set, the script will load the noisy XAI models checkpoints. Default is False."
)
parser.add_argument(
    '--load_denoised_xai',
    action='store_false',
    help="If set, the script will load the denoised XAI models checkpoints. Default is False."
)
parser.add_argument(
    '--load_denoising_method',
    action='store_false',
    help="If set, the script will load the denoising method checkpoints. Default is False."
)
parser.add_argument(
    '--denoising_methods',
    type=str,
    default='["dae", "dlnr", "emd", "kalman_filter", "moving_average", "pca", "resnet", \
        "wavelet_transform"]',
    help="List of denoising methods to use as a JSON string. Default is ['dae', 'dlnr', 'emd', \
        'kalman_filter', 'moving_average', 'pca', 'resnet', 'wavelet_transform']"
)
parser.add_argument(
    '--noise',
    type=float,
    default=0.0,
    help="Noise level to add to the data. Default is 0.0."
)
parser.add_argument(
    '--slurm_id',
    type=str,
    default=None,
    help="Slurm job ID. Default is None."
)
args = parser.parse_args()
args.denoising_methods = json.loads(args.denoising_methods)


# Global variables & Flags #
# ------------------------------------------------------------------------------------------------ #
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'data', 'time_series', args.data_folder)
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'time_series', args.data_folder)
CHECKPOINT_PATH = os.path.join(_CURRENT_DIR, 'checkpoints', 'time_series', args.data_folder)
GRADIENTS_PATH = os.path.join(_CURRENT_DIR, 'gradients', 'time_series', args.data_folder)
if not os.path.exists(GRADIENTS_PATH):
    os.makedirs(GRADIENTS_PATH)

# Flags
VERBOSE = args.verbose # If True, the script will print the progress of the training and testing.
TRAIN_ORIGINAL_XAI = args.load_original_xai
TRAIN_NOISY_XAI = args.load_noisy_xai
TRAIN_DENOISED_XAI = args.load_denoised_xai
TRAIN_DENOISING_METHOD = args.load_denoising_method
IS_CNN = False # If True, the script will take into accoount that a CNN model will be used.

# Device
DEVICE = args.device


# Functions #
# ------------------------------------------------------------------------------------------------ #
def dae(noisy_data: dict, test_size: float = 0.2, random_state: int = 42, batch_size: int = 64,
        latent_dim: int = 8, lr: float = 0.001, criterion: nn.Module = nn.MSELoss(),
        optimizer: optim.Optimizer = optim.Adam, epoch_scheduler: optim.lr_scheduler = None,
        batch_scheduler: optim.lr_scheduler = None, epochs: int = 500, patience: int = 15,
        checkpoint_path: str = None, should_train: bool = True) -> tuple:
    """
    Trains a Denoising Autoencoder (DAE) model to reconstruct noisy input data and evaluates its
    performance.
    Parameters:
        noisy_data (dict): A dictionary containing the noisy dataset. It must include a key 'df'
            with the data as a pandas DataFrame.
        test_size (float, optional): Proportion of the dataset to include in the test split.
            Default is 0.2.
        random_state (int, optional): Random seed for reproducibility of the train-test split.
            Default is 42.
        batch_size (int, optional): Batch size for the DataLoader. Default is 64.
        latent_dim (int, optional): Dimensionality of the latent space in the autoencoder.
            Default is 8.
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
        criterion (nn.Module, optional): Loss function to use during training.
            Default is nn.MSELoss().
        optimizer (optim.Optimizer, optional): Optimizer class to use for training.
            Default is optim.Adam.
        epoch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler to apply at
            the epoch level. Default is None.
        batch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler to apply at
            the batch level. Default is None.
        epochs (int, optional): Number of training epochs. Default is 500.
        patience (int, optional): Number of epochs to wait for improvement before early stopping.
            Default is 15.
        checkpoint_path (str, optional): Path to save or load model checkpoints. Default is None.
        should_train (bool, optional): Whether to train the model or load weights from a checkpoint.
            Default is True.
    Returns:
        tuple: A tuple containing:
            - df_denoised (pd.DataFrame): The denoised dataset as a pandas DataFrame.
            - dae_metrics (dict): A dictionary containing evaluation metrics:
                - 'mse': Mean Squared Error.
                - 'rmse': Root Mean Squared Error.
                - 'mae': Mean Absolute Error.
                - 'smape': Symmetric Mean Absolute Percentage Error.
                - 'R2': R-squared score.
    """
    df_data = noisy_data['df'].copy()

    # Split the data into train and test sets
    train_dae, test_dae = train_test_split(
        df_data.values, test_size=test_size, random_state=random_state
    )

    # The x and y values are the same, as we want to reconstruct the input data
    train_dataset = TensorDataset(
        x=train_dae,
        y=train_dae
    )
    val_dataset = TensorDataset(
        x=test_dae,
        y=test_dae
    )

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create Neural Network model
    model = DenoisingAutoencoder(
        input_dim=train_dae.shape[1],
        latent_dim=latent_dim,
    ).to(DEVICE)

    # Set model parameters and create the model Trainer object
    optimizer = optimizer(model.parameters(), lr=lr)

    # Define the trainer
    trainer_basic = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=val_dataloader,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        batch_scheduler=batch_scheduler,
        patience=patience,
        epochs=epochs,
        checkpoints_path=checkpoint_path
    )

    ## Train the Neural Network or load its weights from a checkpoint ##
    ## -------------------------------------------------------------------------------------- ##
    if os.path.exists(checkpoint_path) and not should_train:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        if VERBOSE:
            print(
                'Checkpoint loaded for the neural network model from path: ',
                checkpoint_path
            )
    else:
        model, _, _, _, _ = trainer_basic.fit(verbose=VERBOSE)

    ## Predict and get the metrics for de NN model ##
    ## ------------------------------------------------------------------------------------------ ##
    df_denoised = model(
        torch.tensor(df_data.values).float().to(DEVICE)
    ).cpu().detach().numpy()
    df_denoised = pd.DataFrame(df_denoised, columns=df_data.columns)

    # Show the metrics
    gt_values = df_data.values
    predicted_values = df_denoised.values
    mae = mean_absolute_error(gt_values, predicted_values)
    smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    dae_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'R2': r_squared
    }

    return df_denoised, dae_metrics


def dlnr(noisy_data: dict,
         batch_size: int = 64, lr: float = 0.001, criterion: nn.Module = nn.MSELoss(),
         optimizer: optim.Optimizer = optim.Adam, epoch_scheduler: str = None,
         batch_scheduler: optim.lr_scheduler = None, epochs: int = 500, patience: int = 15,
         checkpoint_path: str = None, gradients_path: str = None, should_train: bool = True,
         model_params_dict: dict = None, target_var: str = 'y',
         ) -> tuple:
    """
    Perform training, evaluation, and gradient-based denoising using a neural network (NN) model.
    Args:
        noisy_data (dict): A dictionary containing noisy data with the following keys:
            - 'x_train': Training input features (numpy array or tensor).
            - 'y_train': Training target values (numpy array or tensor).
            - 'x_test': Testing input features (numpy array or tensor).
            - 'y_test': Testing target values (numpy array or tensor).
            - 'df': Original noisy dataframe containing input features and target values.
        batch_size (int, optional): Batch size for training and validation. Defaults to 64.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        criterion (nn.Module, optional): Loss function to use during training.
            Defaults to nn.MSELoss().
        optimizer (optim.Optimizer, optional): Optimizer class to use for training.
            Defaults to optim.Adam.
        epoch_scheduler (str, optional): Type of epoch scheduler to use ('StepLR', 'ReduceLROnPlateau', etc.).
            Defaults to None.
        batch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler for batches.
            Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 500.
        patience (int, optional): Early stopping patience for validation loss. Defaults to 15.
        checkpoint_path (str, optional): Path to save or load model checkpoints. Defaults to None.
        gradients_path (str, optional): Path to save the gradients. Defaults to None.
        should_train (bool, optional): Whether to train the model or load from checkpoint.
            Defaults to True.
    Returns:
        tuple: A tuple containing:
            - df_denoised (pd.DataFrame): The denoised dataframe after applying gradient-based
                noise reduction.
            - nn_metrics (dict): A dictionary containing evaluation metrics for the NN model:
                - 'mse': Mean Squared Error.
                - 'rmse': Root Mean Squared Error.
                - 'mae': Mean Absolute Error.
                - 'smape': Symmetric Mean Absolute Percentage Error.
                - 'R2': R-squared score.
    """
    df_data = noisy_data['df'].copy()

    # Split the data into train and test sets
    input_vars = list(set(df_data.columns) - set([target_var]))
    x_train, x_test, y_train, y_test = train_test_split(
        df_data[input_vars].values, df_data[target_var].values, test_size=0.2, random_state=42
    )

    # The x and y values are the same, as we want to reconstruct the input data
    train_dataset = TensorDataset(
        x=x_train,
        y=y_train.reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        x=x_test,
        y=y_test.reshape(-1, 1)
    )

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create Neural Network model
    model = GridFullyDenseNN(**model_params_dict).to(DEVICE)

    # Set model parameters and create the model Trainer object
    optimizer = optimizer(model.parameters(), lr=lr)

    # Define epoch scheduler if specified
    if epoch_scheduler == 'StepLR':
        epoch_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif epoch_scheduler == 'ReduceLROnPlateau':
        epoch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        epoch_scheduler = None

    # Define the trainer
    trainer_basic = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=val_dataloader,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        batch_scheduler=batch_scheduler,
        patience=patience,
        epochs=epochs,
        checkpoints_path=checkpoint_path
    )

    ## Train the Neural Network or load its weights from a checkpoint
    if os.path.exists(checkpoint_path) and not should_train:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        if VERBOSE:
            print(
                'Checkpoint loaded for the neural network model from path: ',
                checkpoint_path
            )
    else:
        model, _, _, _, _ = trainer_basic.fit(verbose=VERBOSE)

    ## Predict and get the metrics for de NN model
    y_pred_test = model(
        torch.tensor(x_test,).float().to(DEVICE)
    ).cpu().detach().numpy().reshape(-1)

    # Show the metrics
    gt_values = y_test
    predicted_values = y_pred_test
    mae = mean_absolute_error(gt_values, predicted_values)
    smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    dlnr_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'R2': r_squared
    }

    if VERBOSE:
        print(f'NN metrics {json.dumps(dlnr_metrics, indent=4)}')

    ## Perform gradient-based denoising method
    save_gradients = gradients_path is not None
    df_denoised = noisy_data['df'].copy()
    input_vars = list(set(df_denoised.columns) - set([target_var]))
    dlnr_model = DLNoiseReduction(
        model=model,
        criterion=criterion,
        is_ts=False,
        is_cnn=IS_CNN,
        device=DEVICE
    )
    dlnr_model.fit(noisy_data['df'][input_vars].values, noisy_data['df'][target_var].values.reshape(-1, 1))
    df_denoised[input_vars], df_denoised[target_var], x_gradients, y_gradients = dlnr_model.transform(
        nrr=0.02,
        nr_threshold=0.02,
        max_epochs=10000,
        plot_progress=False,
        path_to_save_imgs=None,
        save_gradients=save_gradients
    )

    if save_gradients:
        np.save(os.path.join(gradients_path, 'x_gradients.npy'), np.array(x_gradients))
        np.save(os.path.join(gradients_path, 'y_gradients.npy'), np.array(y_gradients))

    return df_denoised, dlnr_metrics


def empirical_mode_decomposition(noisy_data: dict, imgs_to_drop: int = 2) -> pd.DataFrame:
    """
    Perform Empirical Mode Decomposition (EMD) on the input noisy data and reconstruct the signal 
    by summing the Intrinsic Mode Functions (IMFs) after dropping a specified number of
    initial IMFs.
    Parameters:
        noisy_data (dict): A dictionary containing the noisy data. It must have a key 'df' 
                           with a pandas DataFrame as its value, where each column represents 
                           a signal to be denoised.
        imgs_to_drop (int, optional): The number of initial IMFs to drop during reconstruction. 
                                      Defaults to 2.
    Returns:
        dict: A pandas DataFrame (wrapped in a dictionary) containing the denoised signals, 
              where each column corresponds to the denoised version of the respective column 
              in the input DataFrame.
    """
    df_data = noisy_data['df'].copy()
    df_denoised = pd.DataFrame(columns=df_data.columns)

    # Perform EMD on each column of the DataFrame
    for col in df_data.columns:
        emd = EMD()
        imfs = emd(df_data[col].values)
        # Reconstruction of the signal by summing the IMFs
        # after dropping the specified number of initial IMFs
        df_denoised[col] = np.sum(imfs[imgs_to_drop:], axis=0)
    df_denoised = df_denoised.copy()

    return df_denoised


def kalman_filter(noisy_data: dict, transition_matrix: float = 1.0, observation_matrix: float = 1.0,
                  process_noise_variance: float = 1e-5, measurement_noise_variance: float = 1e-2,
                  initial_state: float = 0.0, initial_variance: float = 1.0) -> pd.DataFrame:
    """
    Apply a Kalman Filter to denoise the input data.

    Parameters:
        noisy_data (dict): A dictionary containing the noisy data. It must have a key 'df' 
                           with a pandas DataFrame as its value, where each column represents 
                           a signal to be denoised.
        transition_matrix (float, optional): State transition matrix. Defaults to 1.0.
        observation_matrix (float, optional): Observation matrix. Defaults to 1.0.
        process_noise_variance (float, optional): Process noise covariance. Defaults to 1e-5.
        measurement_noise_variance (float, optional): Measurement noise covariance.
            Defaults to 1e-2.
        initial_state (float, optional): Initial state estimate. Defaults to 0.0.
        initial_variance (float, optional): Initial state covariance. Defaults to 1.0.

    Returns:
        pd.DataFrame: A DataFrame containing the denoised signals, where each column corresponds 
                      to the denoised version of the respective column in the input DataFrame.
    """
    df_data = noisy_data['df'].copy()
    df_denoised = pd.DataFrame(columns=df_data.columns)

    # Kalman Filter Initialization
    for col in df_data.columns:
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[transition_matrix]])  # State transition matrix
        kf.H = np.array([[observation_matrix]])  # Observation matrix
        kf.Q = np.array([[process_noise_variance]])  # Process noise covariance
        kf.R = np.array([[measurement_noise_variance]])  # Measurement noise covariance
        kf.x = np.array([[initial_state]])  # Initial state
        kf.P = np.array([[initial_variance]])  # Initial state covariance

        filtered_signal = []
        for z in df_data[col].values:
            kf.predict()
            kf.update(z)
            filtered_signal.append(kf.x[0, 0])

        df_denoised[col] = filtered_signal

    return df_denoised


def moving_average(noisy_data: dict, window_size: int = 5) -> pd.DataFrame:
    """
    Apply a moving average filter to denoise tabular data.
    This function takes a dictionary containing a DataFrame and applies a moving 
    average filter to each column of the DataFrame. The result is a denoised 
    DataFrame where each column has been smoothed using the specified window size.
    Args:
        noisy_data (dict): A dictionary containing the key 'df', which maps to a 
            pandas DataFrame with the noisy data to be denoised.
        window_size (int, optional): The size of the moving average window. Defaults 
            to 5. A larger window size results in more smoothing.
    Returns:
        pd.DataFrame: A DataFrame containing the denoised data, where each column 
        has been smoothed using the moving average filter.
    """
    df_data = noisy_data['df'].copy()
    df_denoised = pd.DataFrame()

    # Perform moving average denoising
    df_denoised = pd.concat(
        {
            col: df_data[col].rolling(window=window_size, min_periods=1).mean()
            for col in df_data.columns
        },
        axis=1
    )
    return df_denoised


def pca(noisy_data: dict, variance_threshold: float = 0.95) -> pd.DataFrame:
    """
    Perform Principal Component Analysis (PCA) on the given noisy data to reduce dimensionality 
    while retaining a specified amount of variance, and reconstruct the denoised signal.
    Args:
        noisy_data (dict): A dictionary containing the noisy data. The key 'df' should map to 
                           a pandas DataFrame with the input data.
        variance_threshold (float, optional): The minimum cumulative variance ratio to retain 
                                              during dimensionality reduction. Defaults to 0.95.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the denoised data reconstructed from the 
                      reduced principal components.
    """
    df_data = noisy_data['df'].copy()

    # Perform PCA
    pca_model = PCA()
    pca_model.fit(df_data)

    # Select principal components with sufficient variance
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Reduce dimensionality and reconstruct the signal
    pca_denoising = PCA(n_components=n_components)
    data_reduced = pca_denoising.fit_transform(df_data)
    df_denoised = pca_denoising.inverse_transform(data_reduced)
    df_denoised = pd.DataFrame(df_denoised, columns=df_data.columns)

    return df_denoised


def resnet(noisy_data: dict, test_size: float = 0.2, random_state: int = 42, batch_size: int = 64,
        hidden_dim: int = 64, lr: float = 0.001, criterion: nn.Module = nn.MSELoss(),
        optimizer: optim.Optimizer = optim.Adam, epoch_scheduler: optim.lr_scheduler = None,
        batch_scheduler: optim.lr_scheduler = None, epochs: int = 500, patience: int = 15,
        checkpoint_path: str = None, should_train: bool = True) -> tuple:
    """
    Trains a DenseResNetDenoising model to reconstruct noisy tabular data and evaluates its
    performance.
    Args:
        noisy_data (dict): A dictionary containing the noisy data with a key 'df' for the DataFrame.
        test_size (float, optional): Proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        batch_size (int, optional): Number of samples per batch for training. Defaults to 64.
        hidden_dim (int, optional): Number of hidden units in the DenseResNetDenoising model.
            Defaults to 64.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        criterion (nn.Module, optional): Loss function to use during training.
            Defaults to nn.MSELoss().
        optimizer (optim.Optimizer, optional): Optimizer class to use for training.
            Defaults to optim.Adam.
        epoch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler for epochs.
            Defaults to None.
        batch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler for batches.
            Defaults to None.
        epochs (int, optional): Maximum number of training epochs. Defaults to 500.
        patience (int, optional): Number of epochs to wait for improvement before early stopping.
            Defaults to 15.
        checkpoint_path (str, optional): Path to save or load model checkpoints. Defaults to None.
        should_train (bool, optional): Whether to train the model or load from checkpoint.
            Defaults to True.
    Returns:
        tuple: A tuple containing:
            - df_denoised (pd.DataFrame): The denoised DataFrame reconstructed by the model.
            - resnet_metrics (dict): A dictionary containing evaluation metrics:
                - 'mse': Mean Squared Error.
                - 'rmse': Root Mean Squared Error.
                - 'mae': Mean Absolute Error.
                - 'smape': Symmetric Mean Absolute Percentage Error.
                - 'R2': R-squared score.
    """
    df_data = noisy_data['df'].copy()

    # Split the data into train and test sets
    train_nn, test_nn = train_test_split(
        df_data.values, test_size=test_size, random_state=random_state
    )

    # The x and y values are the same, as we want to reconstruct the input data
    train_dataset = TensorDataset(
        x=train_nn,
        y=train_nn
    )
    val_dataset = TensorDataset(
        x=test_nn,
        y=test_nn
    )

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create Neural Network model
    model = DenseResNetDenoising(
        input_dim=train_nn.shape[1],
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    # Set model parameters and create the model Trainer object
    optimizer = optimizer(model.parameters(), lr=lr)

    # Define the trainer
    trainer_basic = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=val_dataloader,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        batch_scheduler=batch_scheduler,
        patience=patience,
        epochs=epochs,
        checkpoints_path=checkpoint_path
    )

    ## Train the Neural Network or load its weights from a checkpoint ##
    ## -------------------------------------------------------------------------------------- ##
    if os.path.exists(checkpoint_path) and not should_train:
        model.load_state_dict(torch.load(checkpoint_path))
        if VERBOSE:
            print(
                'Checkpoint loaded for the neural network model from path: ',
                checkpoint_path
            )
    else:
        model, _, _, _, _ = trainer_basic.fit(verbose=VERBOSE)

    ## Predict and get the metrics for de NN model ##
    ## ------------------------------------------------------------------------------------------ ##
    df_denoised = model(
        torch.tensor(df_data.values).float().to(DEVICE)
    ).cpu().detach().numpy()
    df_denoised = pd.DataFrame(df_denoised, columns=df_data.columns)

    # Show the metrics
    gt_values = df_data.values
    predicted_values = df_denoised.values
    mae = mean_absolute_error(gt_values, predicted_values)
    smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    resnet_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'R2': r_squared
    }

    return df_denoised, resnet_metrics


def wavelet_transform(noisy_data: dict, wavelet_type: str = 'db4') -> pd.DataFrame:
    """
    Apply wavelet transform denoising to the input data.
    This function performs wavelet decomposition on each column of the input 
    DataFrame, applies soft thresholding to the wavelet coefficients to remove 
    noise, and then reconstructs the denoised signal.
    Args:
        noisy_data (dict): A dictionary containing the key 'df', which maps to 
            a pandas DataFrame. Each column of the DataFrame represents a 
            signal to be denoised.
    Returns:
        pd.DataFrame: A DataFrame containing the denoised signals, with the 
        same structure as the input DataFrame.
    Notes:
        - The wavelet used for decomposition and reconstruction is Daubechies 4 ('db4').
        - The threshold for soft thresholding is determined using the universal 
          Donoho rule, with an estimated noise standard deviation (sigma).
        - The function assumes that the input signals are 1-dimensional and 
          contained in the columns of the DataFrame.
    """
    df_data = noisy_data['df'].copy()

    df_denoised = pd.DataFrame(columns=df_data.columns)
    for col in df_data.columns:
        # Decompose the signal
        coeffs = pywt.wavedec(df_data[col], wavelet_type, mode='smooth')

        # Delete the coefficients below a threshold
        ## Sigma is not supposed to be known, but we can estimate it
        epsilon = 1e-10
        sigma_for_wavelet = (np.median(np.abs(coeffs[-1])) + epsilon) / 0.6745
        ## Define threshold by the universal Donoho rule
        threshold = sigma_for_wavelet * np.sqrt(2 * np.log(len(df_data[col])))
        coeffs_denoised = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]

        # Recompose the signal with the last coefficients
        df_denoised[col] = pywt.waverec(coeffs_denoised, wavelet_type)[:len(df_data[col])]

    return df_denoised


# Main function #
# ------------------------------------------------------------------------------------------------ #
def main():
    """
    Main function to execute the tabular experiment pipeline.
    This function sets up the parameters for various XAI models, 
    configures the denoising method, and initializes the TabularExperiment 
    class to run the experiment with specified configurations.
    XAI Models Parameters:
        - ridge: Parameters for Ridge regression model.
        - pls: Parameters for Partial Least Squares regression model.
        - tree: Parameters for Decision Tree model.
        - svm: Parameters for Support Vector Machine model.
        - knn: Parameters for K-Nearest Neighbors model.
    Denoising Method Parameters:
        - batch_size: Batch size for training the denoising method.
        - lr: Learning rate for the optimizer.
        - criterion: Loss function used during training.
        - optimizer: Optimizer used for training.
        - epoch_scheduler: Scheduler for epochs (if any).
        - batch_scheduler: Scheduler for batches (if any).
        - epochs: Number of training epochs.
        - patience: Early stopping patience.
        - checkpoint_path: Path to save/load the model checkpoint.
        - should_train: Boolean indicating whether to train the denoising method.
    Experiment Configuration:
        - data_path: Path to the dataset.
        - out_path: Path to save experiment outputs.
        - checkpoint_path: Path to save/load checkpoints.
        - subfix_name: Subfix for naming outputs.
        - is_cnn: Boolean indicating if CNN is used.
        - train_original_xai: Boolean to train on original XAI data.
        - train_noisy_xai: Boolean to train on noisy XAI data.
        - train_denoised_xai: Boolean to train on denoised XAI data.
        - train_denoising_method: Boolean to train the denoising method.
    The experiment is executed by calling the `run` method of the 
    TabularExperiment class with the specified parameters.
    Parameters:
        None
    Returns:
        None
    """
    # Explainable AI models configuration
    # -------------------------------------------------------------------------------------------- #
    xai_models_parms = {
        'ridge': {"alpha": 1.0},
        'pls': {"n_components": 1},
        'tree': {"max_depth": 5},
        'svm': {"kernel": 'poly', "degree": 2},
        'knn': {
            "n_neighbors": 5,
            "weights": 'uniform',
            "algorithm": 'auto',
            "leaf_size": 30,
            "p": 2,
            "n_jobs": None
        },
        'arima': {
            'order': (7, 0, 0),
            'seasonal_order': (0, 0, 1, 30)
        }
    }

    # Denoising methods configuration
    # -------------------------------------------------------------------------------------------- #
    # DAE - Denoising Autoencoder
    dae_params = {
        'test_size': 0.2,
        'random_state': 42,
        'batch_size': 64,
        'latent_dim': 8,
        'lr': 0.001,
        'criterion': nn.MSELoss(),
        'optimizer': optim.Adam,
        'epoch_scheduler': None,
        'batch_scheduler': None,
        'epochs': 500,
        'patience': 15,
        'checkpoint_path': None,
        'should_train': TRAIN_DENOISING_METHOD,
    }

    data = pd.read_parquet(os.path.join(DATA_PATH, 'clean.parquet'))
    n_var = data.shape[1]-1
    target_var = 'y'
    if 'ETT' in DATA_PATH:
        target_var = 'HULL'

    # DLNR - Deep Learning Noise Reduction
    dlnr_model_params = {
        'n_layers': 10,
        'hidden_layers': [
            (n_var, 64),
            (64, 128),
            (128, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 2048),
            (2048, 1024),
            (1024, 512),
            (512, 128),
            (128, 1)
        ],
        'dropout_layers': [0.3] * 10,
        'activation_func_layers': [nn.ReLU()] * 9 + [nn.Identity()],
        'want_dropout': [False] * 10,
        'want_linear': [True] * 10,
        'want_activation': [True] * 10,
    }
    dlnr_params = {
        'batch_size': 64,
        'lr': 0.01,
        'criterion': nn.MSELoss(),
        'optimizer': optim.Adam,
        'epoch_scheduler': 'ReduceLROnPlateau',  # Initialize as None; will be set dynamically
        'batch_scheduler': None,
        'epochs': 500,
        'patience': 15,
        'checkpoint_path': None,
        # 'gradients_path': GRADIENTS_PATH,
        'should_train': TRAIN_DENOISING_METHOD,
        'model_params_dict': dlnr_model_params,
        'target_var': target_var
    }

    # EMD - Empirical Mode Decomposition
    emd_method_params = {
        'imgs_to_drop': 2
    }

    # Kalman Filter
    kalman_filter_params = {
        'transition_matrix': 1.0,
        'observation_matrix': 1.0,
        'process_noise_variance': 1e-5,
        'measurement_noise_variance': 1e-2,
        'initial_state': 0.0,
        'initial_variance': 1.0
    }

    # MA - Moving Average
    moving_average_params = {
        'window_size': 7
    }

    # PCA - Principal Component Analysis
    pca_params = {
        'variance_threshold': 0.95
    }

    # ResNet - Residual Network
    resnet_params = {
        'test_size': 0.2,
        'random_state': 42,
        'batch_size': 64,
        'hidden_dim': 64,
        'lr': 0.001,
        'criterion': nn.MSELoss(),
        'optimizer': optim.Adam,
        'epoch_scheduler': None,
        'batch_scheduler': None,
        'epochs': 500,
        'patience': 15,
        'checkpoint_path': None,
        'should_train': TRAIN_DENOISING_METHOD
    }

    # Wavelet - Wavelet Transform
    wavelet_transform_params = {
        'wavelet_type': 'db4'
    }

    # General configuration
    full_denoising_methods_dict = {
        'dae': {'method': dae, 'params': dae_params},
        'dlnr': {'method': dlnr, 'params': dlnr_params},
        'emd': {'method': empirical_mode_decomposition, 'params': emd_method_params},
        'kalman_filter': {'method': kalman_filter, 'params': kalman_filter_params},
        'moving_average': {'method': moving_average, 'params': moving_average_params},
        'pca': {'method': pca, 'params': pca_params},
        'resnet': {'method': resnet, 'params': resnet_params},
        'wavelet_transform': {'method': wavelet_transform, 'params': wavelet_transform_params}
    }

    denoising_methods_dict = {k: v for k, v in full_denoising_methods_dict.items()
                              if k in args.denoising_methods}

    for denoising_name, denoising_dict in denoising_methods_dict.items():
        if VERBOSE:
            print(f'Running experiment with denoising method: {denoising_name}')

        # Create the output and checkpoint paths
        current_out_path = os.path.join(OUT_PATH, denoising_name)
        current_checkpoint_path = os.path.join(CHECKPOINT_PATH, denoising_name)

        if denoising_name == 'dlnr':
            denoising_dict['params']['checkpoint_path'] = os.path.join(
                current_checkpoint_path, 'dlnr_model.pt'
            )
        elif denoising_name == 'dae':
            denoising_dict['params']['checkpoint_path'] = os.path.join(
                current_checkpoint_path, 'dae_model.pt'
            )
        elif denoising_name == 'resnet':
            denoising_dict['params']['checkpoint_path'] = os.path.join(
                current_checkpoint_path, 'resnet_model.pt'
            )

        # Create the folders if they do not exist
        if VERBOSE:
            print(f"Creating output path: {current_out_path}")
            print(f"Creating checkpoint path: {current_checkpoint_path}")
        if not os.path.exists(current_out_path):
            os.makedirs(current_out_path)
        if not os.path.exists(current_checkpoint_path):
            os.makedirs(current_checkpoint_path)

        # Experiment configuration
        experiment = TSExperiment(
            data_path=DATA_PATH,
            out_path=current_out_path,
            checkpoint_path=current_checkpoint_path,
            subfix_name=denoising_name,
            is_cnn=IS_CNN,
            train_original_xai=TRAIN_ORIGINAL_XAI,
            train_noisy_xai=TRAIN_NOISY_XAI,
            train_denoised_xai=TRAIN_DENOISED_XAI,
            train_denoising_method=TRAIN_DENOISING_METHOD,
            verbose=VERBOSE,
        )

        if 'synthetic' in DATA_PATH:
            for sigma in np.arange(0.00, 0.16, 0.01):
                sigma = round(sigma, 2)
                # Run the experiment for synthetic data
                experiment.run(
                    data_file=args.data_file,
                    add_noise=True,
                    sigma=sigma,
                    denoising_method=denoising_dict['method'],
                    denoising_method_params=denoising_dict['params'],
                    xai_models_params=xai_models_parms
                )
        else:
            # Run the experiment for real data
            add_noise = args.noise != 0.0

            experiment.run(
                data_file=args.data_file,
                add_noise=add_noise,
                sigma=args.noise,
                denoising_method=denoising_dict['method'],
                denoising_method_params=denoising_dict['params'],
                xai_models_params=xai_models_parms
            )

# Main #
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
    print(f'Running slurm task with id: {args.slurm_id}')
    main()
