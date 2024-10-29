# %%
import os
import sys

CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
assert os.path.exists(LIBS_PATH)
sys.path.append(LIBS_PATH)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List

# Locals
from utils import *
from dataset import Dataset, SlidingWindowDataset
from models import *
from dlnr import DLNoiseReduction

# %%
DATA_PATH = os.path.join(CURRENT_DIR, 'data')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints')
CONFIG_PATH = os.path.join(CURRENT_DIR, 'config')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"

# %% [markdown]
# # Functions

# %%
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

# %% [markdown]
# # Load data

# %%
df = pd.read_csv(os.path.join(DATA_PATH, 'Informer_Datasets', 'WTH', 'WTH.csv'))

# %%
df.index = pd.to_datetime(df['date'])
df.drop(columns=['date'], inplace=True)

# %%
df_small = df[df.columns[1:6]]
df_small['y'] = df[df.columns[-1:]]

# %%
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_small)
df_scaled = pd.DataFrame(df_scaled, columns=df_small.columns)

# %%
sigma = 0.1
df_noisy = add_gaussian_noise(
        df=df_scaled.copy(),
        columns=[x for x in df_scaled.columns],
        mean=0,
        std=sigma
)

# %%
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_noisy[df_noisy.columns[:-1]], df_noisy[df_noisy.columns[-1]], test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

# %%
window_size = 24
train_dataset = SlidingWindowDataset(X_train, y_train, window_size=window_size, future=1)
val_dataset = SlidingWindowDataset(X_val, y_val, window_size=window_size, future=1)
test_dataset = SlidingWindowDataset(X_test, y_test, window_size=window_size, future=1)

# %%
input_size = X_train.shape[-1] * window_size # Número de variables de entrada
hidden_size = 128  # Número de neuronas en la capa oculta
output_size = 1  # Predicción de una variable

# Crear el modelo
model = DenseTemporalModel(input_size, hidden_size, output_size).to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Error cuadrático medio
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# Load checkpoint
# Comment if you want to train the model
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'WTH','dense_sigma0.1.pth')))

# %%
trainer = Trainer(
    model = model,
    train_generator = train_dataloader,
    val_generator = val_dataloader,
    device = device,
    criterion = criterion,
    optimizer = optimizer,
    epoch_scheduler = None,
    batch_scheduler = None,
    patience = 10,
    epochs = 500,
    checkpoints_path = os.path.join(CHECKPOINT_PATH, 'WTH', f'dense_sigma{sigma}'),
)

# %%
predictions = trainer.eval_dataloader(test_dataloader)

# %%
predictions_flattened = np.array([y.cpu().detach().numpy() for x in predictions for y in x])


# %%
test_to_compare = y_test.values[window_size:]

# %%
mse = mean_squared_error(test_to_compare, predictions_flattened)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_to_compare, predictions_flattened)
r2 = r2_score(test_to_compare, predictions_flattened)

# %%
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R2: {r2:.4f}')

# %% [markdown]
# # Gradients to reduce noise

# %%
x_cols = df_noisy.columns[:-1]
df_to_denoise = SlidingWindowDataset(df_noisy[x_cols], df_noisy[['y']], window_size=window_size, future=1)
# df_to_denoise = DataLoader(df_to_denoise, batch_size=len(df_to_denoise), shuffle=False)

# %%
dlnr = DLNoiseReduction(model=model, criterion=criterion, is_ts=True)
dlnr.fit(df_to_denoise)

# %%
X_denoised, y_denoised = dlnr.transform(
    nrr=0.05,
    nr_threshold=0.001,
    max_epochs=500,
    plot_progress=False
)

X_denoised.to_parquet(os.path.join(DATA_PATH, 'Informer_Datasets', 'WTH', 'X_denoised.parquet'))
y_denoised.to_parquet(os.path.join(DATA_PATH, 'Informer_Datasets', 'WTH', 'y_denoised.parquet'))