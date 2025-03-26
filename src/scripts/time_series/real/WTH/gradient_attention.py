# pylint: disable=import-error
# pylint: disable=wrong-import-position
"""
title: real_wth_exp_attention
author: José Javier Alonso Ramos
email: jjalonso@ugr.es
institution: DaSCI - UGR

Description:
Performs a real experiment with a WTH dataset using an attention-based model.
The experiment consists of generating a WTH dataset with a polinomial function and
adding Gaussian noise to it. Then, a neural network model with attention mechanisms
is trained to predict the target variable. Finally, the gradients are used to reduce
the noise in the data.
"""

# Libraries & Global variables #
# ------------------------------------------------------------------------------------------------ #
# Public libraries
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global variables
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'time_series', 'real', 'WTH')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', 'time_series', 'real', 'WTH')
OUT_PATH = os.path.join(CURRENT_DIR, 'out', 'time_series', 'real', 'WTH')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# Show info on the terminal about how the execution is going.
VERBOSE = True
# Even if there is a checkpoint, the model is retrained.
FORCE_TRAINING_PRE_XAI = True
FORCE_TRAINING_NN = True
FORCE_TRAINING_POST_XAI = True
# Name of this experiment that will appear in the result files.
SUBFIX_NAME = 'gradient_attention'
IS_TS = True

# Local libraries
from dataset import SlidingWindowDataset
from models import Trainer, XAI_benchmark
from dlnr import DLNoiseReduction
from utils import symmetric_mean_absolute_percentage_error
from TSFEDL.blocks_pytorch import TemporalAttentionBlockZhangJin, SpatialAttentionBlockZhangJin, SqueezeAndExcitationModule, RTABlock

# Make sure that the GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"


# Model definition #
# ------------------------------------------------------------------------------------------------ #
class WTHModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(WTHModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa convolucional inicial para capturar patrones locales
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Mecanismo de atención temporal
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas fully connected con skip connection
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.size()
        
        # Aplicar capa convolucional
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)
        
        # LSTM bidireccional
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Mecanismo de atención
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Capas fully connected con skip connection
        out = self.fc1(context)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class WTHModelTSFEDL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WTHModelTSFEDL, self).__init__()
        
        # Bloque de atención temporal de Zhang-Jin
        self.temporal_attention = TemporalAttentionBlockZhangJin()
        
        # Bloque RTA para capturar patrones temporales
        self.rta = RTABlock(in_features=input_size, nb_filter=hidden_size, kernel_size=3)
        
        # Bloque de atención espacial
        self.spatial_attention = SpatialAttentionBlockZhangJin(
            in_features=hidden_size, 
            decrease_ratio=4
        )
        
        # Squeeze and Excitation para calibrar características
        self.se = SqueezeAndExcitationModule(
            in_features=hidden_size, 
            dense_units=hidden_size//4
        )
        
        # Capa final de predicción
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Aplicar atención temporal
        x = self.temporal_attention(x)
        
        # Aplicar RTA block
        x = self.rta(x)
        
        # Aplicar atención espacial
        x = self.spatial_attention(x)
        
        # Aplicar SE
        x = self.se(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)  # (batch, features)
        
        # Predicción final
        x = self.fc(x)
        
        return x


# Functions definition #
# ------------------------------------------------------------------------------------------------ #
def dictionary_arrays_to_list(array_d:dict):
    """
    Run through a array_d dictionary converting its arrays to lists.

    Args:
        array_d (dict): dictionary which arrays we want to convert to lists.

    Returns:
        dict: dictionary with lists intead of arrays.
    """
    if isinstance(array_d, dict):
        return {k: dictionary_arrays_to_list(v) for k, v in array_d.items()}
    elif isinstance(array_d, np.ndarray):
        return array_d.tolist()
    elif isinstance(array_d, pd.Series):
        return array_d.to_list()
    else:
        return array_d


# Main #
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
    ## Load the data ##
    ## ------------------------------------------------------------------------------------------ ##
    predictions_dict = {}
    metrics_dict = {}
    predictions_dict['orig'] = {}
    metrics_dict['orig'] = {}

    # Load the data
    df_data = pd.read_parquet(
        os.path.join(
            DATA_PATH,
            'clean.parquet'
        )
    )

    df_data.rename(columns={"Visibility": "y"}, inplace=True)

    # scale the data
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

    # assert there is no more categorical variables
    columnas_categoricas = df_data.select_dtypes(include=['object', 'category']).columns
    assert not list(columnas_categoricas)
    
    input_vars = df_data.columns
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        df_data[input_vars].values, df_data['y'].values, test_size=0.2, shuffle=False
    )


    # Create the sliding window datasets
    window_size = 24  # 24 horas
    target_size = 1   # Predicción a 1 hora
    batch_size = 256

    # Create sliding window datasets
    train_dataset = SlidingWindowDataset(
        X_train_nn,
        y_train_nn,
        window_size=window_size,
        future=1
    )
    val_dataset = SlidingWindowDataset(
        X_test_nn,
        y_test_nn,
        window_size=window_size,
        future=1
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create Neural Network model
    input_size = X_train_nn.shape[1]  # Número de variables de entrada
    hidden_size = 128  # Número de neuronas en la capa oculta
    output_size = 1  # Predicción de una variable
    model = WTHModel(input_size, hidden_size, output_size).to(device)

    # Set model parameters and create the model Trainer object
    lr = 0.0005
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    denoiser_checkpoint_path = os.path.join(
        CHECKPOINT_PATH,
        'orig',
        'nn_orig_attention.pth'
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        train_generator=train_dataloader,
        val_generator=val_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        epoch_scheduler=scheduler,
        batch_scheduler=None,
        patience=10,
        epochs=100,
        checkpoints_path=denoiser_checkpoint_path
    )
    # Define the trainer
    # trainer = Trainer(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=device,
    #     checkpoint_path=denoiser_checkpoint_path,
    #     verbose=VERBOSE
    # )

    # Train the model
    if FORCE_TRAINING_NN or not os.path.exists(denoiser_checkpoint_path):
        model, _, _, _, _ = trainer.fit(verbose=VERBOSE)
    else:
        model.load_state_dict(torch.load(denoiser_checkpoint_path, weights_only=True))
        if VERBOSE:
            print(
                'Checkpoint loaded for the neural network model from path: ',
                denoiser_checkpoint_path
            )

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions_test = trainer.eval_dataloader(val_dataloader)
        y_pred_test = np.array([y for x in predictions_test for y in x])

    # Convert predictions to numpy arrays
    y_pred_test = np.array(y_pred_test)

    # Inverse transform the predictions
    y_pred_test_orig = scaler.inverse_transform(np.concatenate([X_test_nn[window_size:], y_pred_test], axis=1))[:, -1]

    # Calculate metrics
    gt_values = y_test_nn[window_size:]
    predicted_values = y_pred_test_orig
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

    predictions_dict['orig'][SUBFIX_NAME] = predicted_values
    metrics_dict['orig'][SUBFIX_NAME] = nn_metrics

    # Save results
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # Save the predictions
    with open(os.path.join(OUT_PATH, f'predictions_{SUBFIX_NAME}.json'), 'w') as f:
        json.dump(dictionary_arrays_to_list(predictions_dict), f)

    # Save the metrics
    with open(os.path.join(OUT_PATH, f'metrics_{SUBFIX_NAME}.json'), 'w') as f:
        json.dump(dictionary_arrays_to_list(metrics_dict), f)

    print("Experimento completado con éxito.")
    print(f"Métricas del modelo: {nn_metrics}") 