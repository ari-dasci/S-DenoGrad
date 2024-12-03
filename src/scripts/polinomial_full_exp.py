# # Description
# 
# Prove that the Noise Reduction model works on a 2D polinomial.

# Add libs folder path to system variables to make the importation of local libraries easier.
# __________________________________________________________________________________________________
import os
import sys


CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)


# Import external and local libraries
# __________________________________________________________________________________________________
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import json
from typing import List

# Locals
from utils import *
from dataset import Dataset
from models import *
import config as cfg
from dlnr import DLNoiseReduction


# Global / Path variables
# __________________________________________________________________________________________________
DATA_PATH = os.path.join(CURRENT_DIR, 'data')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')


# Make sure that the GPU is being used
# __________________________________________________________________________________________________
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"


# Functions definition
# __________________________________________________________________________________________________
def polinomial_function(x:float):
    """
    Function that takes in a value x and returns its polinomial value.

    Args:
        x (float): value to be transformed.

    Returns:
        float: result of the polinomial function.
    """

    return x**4 -x**3 -20*(x**2) -20*x +6


def add_gaussian_noise(df:pd.DataFrame, columns:List[str], mean:float = 0.0, std:float = 0.1):
    """
    Adds Gaussian noise to specified columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to which noise will be added.
        columns (list): A list of column names in the DataFrame to which noise will be added.
        mean (int, optional): The mean of the Gaussian distribution. Defaults to 0.
        std (float, optional): The standard deviation of the Gaussian distribution. Defaults to 0.1.

    Returns:
        pandas.DataFrame: The DataFrame with added Gaussian noise to specified columns.
    """
    for col in columns:
        df[col] = df[col] + np.random.normal(loc=mean, scale=std, size=len(df))
    return df


# Main
# __________________________________________________________________________________________________
# Generate continious 2D data
X_train = np.linspace(-5, 5, 10001)
y_train = polinomial_function(X_train)
df_train = pd.DataFrame({'X': X_train, 'y': y_train})

# Scale the noisy dataframe into [0,1] domain
scaler = MinMaxScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=['X', 'y'])

model_params = {
    'ridge': {"alpha": 1.0},
    'pls': {"n_components": 1},
    'tree': {"max_depth": 5},
    'svm': {"dual": 'auto'},
    'knn': {
        "n_neighbors": 5,
        "weights": 'uniform',
        "algorithm": 'auto',
        "leaf_size": 30,
        "p": 2,
        "n_jobs": None
    },
}
xai_benchmark = XAI_benchmark(is_ts = False, model_params = model_params)

json_content = {}
for sig in np.arange(0,0.16,0.01):
    # Add goussian noise to the dataframe
    sigma = sig
    df_noisy = add_gaussian_noise(
            df=df_train.copy(),
            columns=[x for x in df_train.columns],
            mean=0.0,
            std=sigma
    )
    # Noisy XAI Benchmarck
    xai_benchmark.fit(df_noisy[['X']], df_noisy['y'].values)
    noisy_predictions, noisy_benchmark = xai_benchmark.predict(
        df_noisy[['X']],
        df_train['y'].values,
        get_metrics=True
    )

    # Create Datasets for DataLoaders
    noisy_dataset = Dataset(
        input_df=df_noisy[['X']],
        target_df=df_noisy[['y']]
    )

    # Create the dataloaders
    batch_size = 64
    train_dataloader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create Neural Network model
    model = GridFullyDenseNN(
        n_layers=2,
        hidden_layers=[
            (1, 1024),
            (1024, 1),
        ],
        dropout_layers=[
            0.0,
            0.0,
        ],
        activation_func_layers=[
            nn.ReLU(),
            nn.Identity(),
        ],
        want_dropout=[
            False,
            False,
        ],
        want_linear=[
            True,
            True,
        ],
        want_activation=[
            True,
            False,
        ],
    )
    model.to(device)

    # Set model parameters and create the model Trainer object
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the trainer
    trainer_basic = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=train_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=None,
        batch_scheduler=None,
        patience=15,
        epochs=500,
        checkpoints_path=os.path.join(CHECKPOINT_PATH, f'model_noise_{sigma}_polinomial')
    )

    # Train the model
    model, train_losses, val_losses, best_train_loss, best_val_loss = trainer_basic.fit(verbose=True)
    y_pred_noisy = model(
        torch.tensor(
            df_noisy['X'].values.reshape(-1, 1)
        ).float().to(device)
    ).cpu().detach().numpy().reshape(-1)

    # Show the metrics
    gt_values = df_train['y'].values
    predicted_values = y_pred_noisy

    mae = mean_absolute_error(gt_values, predicted_values)
    mse = mean_squared_error(gt_values, predicted_values)
    rmse = np.sqrt(mse)
    # mape = np.mean(np.abs(gt_values - predicted_values) / gt_values) * 100
    r_squared = r2_score(gt_values, predicted_values)
    nn_benchmark = {
        'nn': {
            "MSE: ": mse,
            "RMSE: ": rmse,
            "MAE: ": mae,
            # "MAPE: ": mape,
            "R2: ": r_squared
        }
    }

    # Gradients to reduce noise
    df_no_noise = df_noisy.copy()
    dlnr = DLNoiseReduction(model=model, criterion=criterion)
    dlnr.fit(df_noisy['X'].values.reshape(-1, 1), df_noisy['y'].values.reshape(-1, 1))

    df_no_noise['X'], df_no_noise['y'] = dlnr.transform(
        nrr=0.05,
        nr_threshold=0.01,
        max_epochs=200,
        plot_progress=False
    )

    # Denoised XAI Benchmark
    xai_benchmark.fit(df_no_noise[['X']], df_no_noise['y'].values)
    denoised_predictions, denoised_benchmark = xai_benchmark.predict(
        df_no_noise[['X']],
        df_train['y'].values,
        get_metrics=True
    )

    json_content[f'sigma={sig}'] = {
        'noisy': noisy_benchmark,
        'nn': nn_benchmark,
        'denoised': denoised_benchmark
    }

with open('logs/polinomial_results.json', 'w') as jf:
    json.dump(json_content, jf, ensure_ascii=False, indent=4)
