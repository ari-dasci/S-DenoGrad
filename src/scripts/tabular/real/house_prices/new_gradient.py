# pylint: disable=import-error
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

# Libraries & Global variables #
# ------------------------------------------------------------------------------------------------ #
# Public libraries
import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# Local libraries
from src.libs.models import Trainer
from src.libs.utils import symmetric_mean_absolute_percentage_error
from src.libs.dataset import TensorDataset
from src.libs.models import GridFullyDenseNN
from src.libs.dlnr import DLNoiseReduction
from src.libs.experiment_templates import TabularExperiment

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SUBFIX_NAME = 'gradient' # Name of this experiment that will appear in the result files.

# Global variables
_CURRENT_DIR = os.getcwd()
_FOLDERS = _CURRENT_DIR.split(os.sep)
_PROJECT_FOLDER_INDEX = _FOLDERS.index('S-noise-gradient')
_CURRENT_DIR = os.sep.join(_FOLDERS[:_PROJECT_FOLDER_INDEX+1])
DATA_PATH = os.path.join(_CURRENT_DIR, 'data', 'tabular', 'real', 'house_prices')
OUT_PATH = os.path.join(_CURRENT_DIR, 'out', 'tabular', 'real', 'house_prices', SUBFIX_NAME)
CHECKPOINT_PATH = os.path.join(_CURRENT_DIR, 'checkpoints', 'tabular', 'real', 'house_prices',
                               SUBFIX_NAME)

# Create the folders if they do not exist
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Flags
VERBOSE = True # If True, the script will print the progress of the training and testing.
TRAIN_ORIGINAL_XAI = False # As working with real data, we do not have the original data.
TRAIN_NOISY_XAI = True # If True, the script will train the XAI models over the noisy data.
TRAIN_DENOISED_XAI = True # If True, the script will train the XAI models over the denoised data.
TRAIN_DENOISING_METHOD = False # If True, the script will train the denoising method.
IS_CNN = False # If True, the script will use a CNN model.

# Make sure that the GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"


def dlnr_method(noisy_data: dict,
                batch_size: int = 64, lr: float = 0.001, criterion: nn.Module = nn.MSELoss(),
                optimizer: optim.Optimizer = optim.Adam, epoch_scheduler: optim.lr_scheduler = None,
                batch_scheduler: optim.lr_scheduler = None, epochs: int = 500, patience: int = 15,
                checkpoint_path: str = None, should_train: bool = True) -> None:
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
        epoch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler for epochs.
            Defaults to None.
        batch_scheduler (optim.lr_scheduler, optional): Learning rate scheduler for batches.
            Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 500.
        patience (int, optional): Early stopping patience for validation loss. Defaults to 15.
        checkpoint_path (str, optional): Path to save or load model checkpoints. Defaults to None.
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
    Notes:
        - The method trains a fully connected NN model using the provided noisy data.
        - If a checkpoint exists and `should_train` is False, the model weights are loaded from
            the checkpoint.
        - After training, the method evaluates the model on the test set and computes various
            metrics.
        - Finally, the method applies a gradient-based noise reduction technique to denoise the
            input data.
    """
    train_dataset = TensorDataset(
        x=noisy_data['x_train'],
        y=noisy_data['y_train'].reshape(-1,1)
    )
    val_dataset = TensorDataset(
        x=noisy_data['x_test'],
        y=noisy_data['y_test'].reshape(-1,1)
    )

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create Neural Network model
    model = GridFullyDenseNN(
        n_layers=7,
        hidden_layers=[
            (noisy_data['x_train'].shape[1], 64),
            (64, 256),
            (256, 1024),
            (1024, 1024),
            (1024, 512),
            (512, 128),
            (128, 1)
        ],
        dropout_layers=[0.0] * 7,
        activation_func_layers=[nn.ReLU()] * 6 + [nn.Identity()],
        want_dropout=[False] * 7,
        want_linear=[True] * 7,
        want_activation=[True] * 7,
    ).to(device)

    # Set model parameters and create the model Trainer object
    optimizer = optimizer(model.parameters(), lr=lr)

    # Define the trainer
    trainer_basic = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=val_dataloader,
        device=device,
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
    ## -------------------------------------------------------------------------------------- ##
    y_pred_test = model(
        torch.tensor(noisy_data['x_test'],).float().to(device)
    ).cpu().detach().numpy().reshape(-1)

    # Show the metrics
    gt_values = noisy_data['y_test']
    predicted_values = y_pred_test
    mae = mean_absolute_error(gt_values, predicted_values)
    smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    r_squared = r2_score(gt_values, predicted_values)

    nn_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'R2': r_squared
    }

    ## Perform gradient-based denoising method ##
    ## -------------------------------------------------------------------------------------- ##
    df_denoised = noisy_data['df'].copy()
    input_vars = list(set(df_denoised.columns) - set(['y']))
    dlnr = DLNoiseReduction(model=model, criterion=criterion)
    dlnr.fit(noisy_data['df'][input_vars].values, noisy_data['df']['y'].values.reshape(-1, 1))
    df_denoised[input_vars], df_denoised['y'] = dlnr.transform(
        nrr=0.05,
        nr_threshold=0.01,
        max_epochs=200,
        plot_progress=False,
        path_to_save_imgs=None
    )

    return df_denoised, nn_metrics


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
    }

    dlnr_checkpoint_path = os.path.join(
        CHECKPOINT_PATH,
        'nn.pth'
    )

    denoising_method = dlnr_method
    denoising_method_params = {
        'batch_size': 64,
        'lr': 0.001,
        'criterion': nn.MSELoss(),
        'optimizer': optim.Adam,
        'epoch_scheduler': None,
        'batch_scheduler': None,
        'epochs': 500,
        'patience': 15,
        'checkpoint_path': dlnr_checkpoint_path,
        'should_train': TRAIN_DENOISING_METHOD
    }

    experiment = TabularExperiment(
        data_path=DATA_PATH,
        out_path=OUT_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        subfix_name=SUBFIX_NAME,
        is_cnn=IS_CNN,
        train_original_xai=TRAIN_ORIGINAL_XAI,
        train_noisy_xai=TRAIN_NOISY_XAI,
        train_denoised_xai=TRAIN_DENOISED_XAI,
        train_denoising_method=TRAIN_DENOISING_METHOD,
    )
    experiment.run(
        data_file='clean.parquet',
        add_noise=False,
        denoising_method=denoising_method,
        denoising_method_params=denoising_method_params,
        xai_models_params=xai_models_parms
    )

# Main #
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
    main()
