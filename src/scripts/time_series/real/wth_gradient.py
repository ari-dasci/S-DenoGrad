import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global path variables
CURRENT_DIR = os.getcwd()
FOLDERS = CURRENT_DIR.split(os.sep)
TESIS_FOLDER_INDEX = FOLDERS.index('S-noise-gradient')
CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'time_series', 'real', 'WTH')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', 'temp', 'WTH')
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
SUBFIX_NAME = 'gradient'
IS_TS = True

# Local libraries
from dataset import SlidingWindowDataset
from models import Trainer, LSTMModel
from dlnr import DLNoiseReduction
from utils import symmetric_mean_absolute_percentage_error
import TSFEDL.models_pytorch as nacho
from TSFEDL.blocks_pytorch import TemporalAttentionBlockZhangJin, SpatialAttentionBlockZhangJin, SqueezeAndExcitationModule, RTABlock

# Make sure that the GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"

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

df_data = pd.read_parquet(
    os.path.join(
        DATA_PATH,
        'clean.parquet'
        # 'clean_h1.parquet' # ETT
    )
)

# df_data.rename(columns={"meantemp": "y"}, inplace=True) # daily_climate
# df_data.rename(columns={"MT_320": "y"}, inplace=True) # ECL
# df_data.rename(columns={"OT": "y"}, inplace=True) # ETT
df_data.rename(columns={"Visibility": "y"}, inplace=True) # WTH

# scale the data
scaler = MinMaxScaler()
df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

# assert there is no more categorical variables
columnas_categoricas = df_data.select_dtypes(include=['object', 'category']).columns
assert not list(columnas_categoricas)

## Declare a Neural Network model and prepare the data to train it ##
## ------------------------------------------------------------------------------------------ ##
# divide the data into train/test datasets
input_vars = df_data.columns
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    df_data[input_vars].values, df_data['y'].values, test_size=0.2, shuffle=False
)

# Create the dataloaders
batch_size = 256
window_size = 48
train_dataset = SlidingWindowDataset(
    X_train_nn,
    y_train_nn,
    window_size=window_size,
    future=1,
    cnn=False
)
val_dataset = SlidingWindowDataset(
    X_test_nn,
    y_test_nn,
    window_size=window_size,
    future=1,
    cnn=False
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create Neural Network model
input_size = X_train_nn.shape[1] # Número de variables de entrada
hidden_size = 128  # Número de neuronas en la capa oculta
output_size = 1  # Predicción de una variable

# model = LSTMModel(input_size, hidden_size, output_size).to(device)

# top_module = nacho.OhShuLih_Classifier(20, 1)
# model = nacho.OhShuLih(
#     in_features=input_size,
#     top_module=top_module,
#     loss=nn.MSELoss()
# ).to(device)

model = WTHModel(input_size, hidden_size, output_size).to(device)

# Set model parameters and create the model Trainer object
lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
denoiser_checkpoint_path = os.path.join(
    CHECKPOINT_PATH,
    'NN.pth'
)

# Define the trainer
trainer_basic = Trainer(
    model=model,
    train_generator=train_dataloader,
    val_generator=val_dataloader,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    epoch_scheduler=scheduler,
    batch_scheduler=None,
    patience=15,
    epochs=500,
    checkpoints_path=denoiser_checkpoint_path
)
# Compactar los pesos del modelo LSTM antes de la evaluación
if hasattr(model, 'lstm'):
    trainer_basic.model.lstm.flatten_parameters()
    trainer_basic.best_model.lstm.flatten_parameters()

## Train the Neural Network or load its weights from a checkpoint ##
## ------------------------------------------------------------------------------------------ ##
if os.path.exists(denoiser_checkpoint_path) and not FORCE_TRAINING_NN:
    model.load_state_dict(torch.load(denoiser_checkpoint_path, weights_only=True))
    if VERBOSE:
        print(
            'Checkpoint loaded for the neural network model from path: ',
            denoiser_checkpoint_path
        )
else:
    model, _, _, _, _ = trainer_basic.fit(verbose=VERBOSE)

# Compactar los pesos del modelo LSTM antes de la evaluación
if hasattr(model, 'lstm'):
    trainer_basic.model.lstm.flatten_parameters()
    trainer_basic.best_model.lstm.flatten_parameters()

complete_dataset = SlidingWindowDataset(
    df_data[input_vars].values,
    df_data['y'].values,
    window_size=window_size,
    future=1,
    cnn=False
)
complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False)
predictions_test = trainer_basic.eval_dataloader(val_dataloader)
y_pred_test = np.array([y for x in predictions_test for y in x])

# Show the metrics
gt_values = y_test_nn[window_size:] #y_test_nn[window_size:]
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

print(nn_metrics)

[markdown]
# - **OhShuLih(Sigmoid):** {'mse': 0.013923307936731466,
#  'rmse': 0.11799706749208416,
#  'mae': 0.04739180313515898,
#  'smape': 11.61509062732754,
#  'R2': 0.45049543248479507}
# 
# - **OhShuLih(ReLU):** {'mse': 0.02337899745714202,
#  'rmse': 0.15290192103810213,
#  'mae': 0.12958510968008316,
#  'smape': 19.25029689696666,
#  'R2': 0.07731223463540393}
# 
# - **KhanZulfiqar(ReLU):** {'mse': 0.026051158771344562,
#  'rmse': 0.1614037136231523,
#  'mae': 0.09285865916837853,
#  'smape': 11.830774220942475,
#  'R2': -0.028148684132182522}
# 
# - **KhanZulfiqar(Sigmoid):** {'mse': 0.026033130562538247,
#  'rmse': 0.16134785577297966,
#  'mae': 0.09255783884810385,
#  'smape': 11.798041527700146,
#  'R2': -0.027437173395782555}
# 
# - **WangKejun(ELU):** {'mse': 0.014995688630142856,
#  'rmse': 0.12245688478049267,
#  'mae': 0.05095952773686952,
#  'smape': 11.66577748690614,
#  'R2': 0.4081722940594722}
# 
# - **WangKejun(Sigmoid):** {'mse': 0.025345289736742008,
#  'rmse': 0.15920204061739288,
#  'mae': 0.06671735851217105,
#  'smape': 9.027834685248166,
#  'R2': -0.00029048690317634573}


