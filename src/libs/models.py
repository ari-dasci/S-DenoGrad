# pylint: disable=too-many-arguments
# -*- coding: utf-8 -*-
"""
Model classes for the BioFuel project.

Description:
    This file contains classes for models used in the BioFuel project. The models
    are implemented using PyTorch. The models are organized into classes to make
    them easy to use and modify. The classes are organized as follows:
        - StackedLSTM: LSTM model with multiple layers.
        - LSTM_FC: LSTM model with fully connected layers.
        - GeneralRegression: Layers to transform any TSFEDL model into a regression model.
        - FullyDenseNN: Fully connected neural network.
        - GridFullyDenseNN: Fully connected neural network with a grid architecture.
        - v2DataModel: Neural network architecture for the best model found for the v2 dataset.
        - v2DataModel_2Heads: Neural network architecture for the best model found for the v2 dataset
            modified to work with v3 dataset adding a second head. Used for transfer learning.
        - v2DataModel_1Head: Neural network architecture for the best model found for the v2 dataset
            modified to work with v3 dataset. Used for transfer learning.
        - v3DataModel: Neural network architecture for the best model found for the v2 dataset
            modified to work with v3 dataset.
        - v3DataModel_Gfradients: Neural network architecture for the best model found for the v2 dataset
            modified to work with v3 dataset. Used for transfer learning.
        - Trainer: Class to perform model training in the style of PyTorch Lightning.
"""
# Add libs path to sys path and create some global path variables
# ---------------------------------------------------------------------------- #
import os

# Libraries
# ---------------------------------------------------------------------------- #
import copy
import pickle
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# XAI models
from sklearn.linear_model import Ridge                  # Ridge regression
from sklearn.cross_decomposition import PLSRegression   # Partial least squares regression
from sklearn.tree import DecisionTreeRegressor          # Decision tree regression
from sklearn.svm import SVR                             # Linear support vector regression
from sklearn.neighbors import KNeighborsRegressor       # K-neighbors regression
from pmdarima import ARIMA                              # ARIMA
from pmdarima import auto_arima                         # Auto ARIMA

# Locals
from config import Colors
from utils import *


# Classes
# ---------------------------------------------------------------------------- #

# Fully connected models
# ---------------------------------------------------------------------------- #
class GridFullyDenseNN(nn.Module):
    """
    Fully connected neural network.

    Args:
        nn (Class): Inherited class.
    """
    layer_names = ['dropout', 'linear', 'activation']
    layer_types = [nn.Dropout, nn.Linear]

    def __init__(self,
                 n_layers:int,
                 hidden_layers:list,
                 dropout_layers:list,
                 activation_func_layers:list,
                 want_dropout:list,
                 want_linear:list,
                 want_activation:list):
        """
        Initializes the model with the input parameters. All parameters that are lists
        should have the same length and match the number of layers in the model (n_layers).

        Args:
            n_layers (int): Number of layers in the model
            hidden_layers (list): List of tuples with the input and output sizes of the layers
            dropout_layers (list): List of dropout probabilities
            activation_func_layers (list): List of activation functions
        """
        # Check that all lists have the same length
        assert n_layers == len(hidden_layers) == len(dropout_layers) == len(activation_func_layers)
        super().__init__()

        # Attributes
        self.n_layers = n_layers
        self.model = nn.Sequential()

        # Add layers
        for i in range(self.n_layers):
            for j, layer_name in enumerate(self.layer_names):
                if layer_name == 'dropout' and want_dropout[i]:
                    self.model.add_module(
                        f'{layer_name}_{i}',
                        self.layer_types[j](p=dropout_layers[i])
                    )
                elif layer_name == 'linear' and want_linear[i]:
                    self.model.add_module(
                        f'{layer_name}_{i}',
                        self.layer_types[j](hidden_layers[i][0], hidden_layers[i][1], bias=True)
                    )
                elif layer_name == 'activation' and want_activation[i]:
                    self.model.add_module(
                        f'{layer_name}_{i}',
                        activation_func_layers[i]
                    )

    def forward(self, x):
        """
        Default method when calling the model().

        Args:
            x (np.array): Input data for inference.

        Returns:
            float: Prediction value.
        """
        return self.model(x)


class DenseTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenseTemporalModel, self).__init__()

        # Capas del modelo
        self.flatten = nn.Flatten()  # Aplana las dimensiones (window, n_features) a (window * n_features,)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primera capa densa
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Segunda capa densa
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Segunda capa densa
        self.fc3 = nn.Linear(hidden_size, output_size)  # Capa de salida

        # Función de activación
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.tensor): Noisy input data.

        Returns:
            torch.tensor: Denoised data.
        """
        # x tiene forma (batch_size, window, n_features)
        x = self.flatten(x)  # Aplana a (batch_size, window * n_features)
        x = self.sigmoid(self.fc1(x))  # Primera capa densa con Sigmoid
        x = self.sigmoid(self.fc2(x))  # Segunda capa densa con Sigmoid
        x = self.fc3(x)  # Capa de salida
        return x


# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # Definir la capa GRU
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

#         # Definir la capa de salida
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
            # """
            # Forward pass of the model.

            # Args:
            #     x (torch.tensor): Noisy input data.

            # Returns:
            #     torch.tensor: Denoised data.
            # """
#         # Estado oculto inicial
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

#         # Paso a través de la GRU
#         out, _ = self.gru(x, h0)

#         # Paso a través de la capa totalmente conectada
#         out = self.fc(out[:, -1, :])  # Solo queremos la salida del último timestep
#         return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Definir la capa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Definir la capa de salida
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.tensor): Noisy input data.

        Returns:
            torch.tensor: Denoised data.
        """
        # Estado oculto inicial
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Paso a través de la LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Paso a través de la capa totalmente conectada
        out = self.fc(out[:, -1, :])  # Solo queremos la salida del último timestep
        return out


class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder model.

    Args:
        nn (torch.module): Inherited class.
    """
    def __init__(self, input_dim, latent_dim):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Para valores normalizados entre 0 y 1
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.tensor): Noisy input data.

        Returns:
            torch.tensor: Denoised data.
        """
        # Forward pass
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoisingAutoencoder_v2(nn.Module):
    def __init__(self, input_dim, hidden_dims, encoding_dim, dropout_rate=0.2,
                 activation=nn.ReLU, output_activation=nn.Sigmoid):
        super(DenoisingAutoencoder_v2, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Build the encoder network
        encoder_layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            out_dim = int(in_dim / 2) if i > 0 else out_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.BatchNorm1d(out_dim))
            encoder_layers.append(activation())
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder network
        decoder_layers = []
        in_dim = encoding_dim
        for i, out_dim in enumerate(reversed(hidden_dims)):
            out_dim = int(in_dim / 2) if i > 0 else out_dim
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.BatchNorm1d(out_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        decoder_layers.append(output_activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, add_noise=True):
        if add_noise:
            x = x + torch.randn_like(x) * 0.1
        encoding = self.encoder(x)
        reconstructed = self.decoder(encoding)
        return reconstructed


class TemporalDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(TemporalDenoisingAutoencoder, self).__init__()
        # Codificador
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduce longitud temporal
        )
        # Decodificador
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=input_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
    
    def forward(self, x):
        # Cambiar dimensiones de (batch, seq_len, feature_dim) a (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Volver a la forma original (batch, seq_len, feature_dim)
        decoded = decoded.permute(0, 2, 1)
        return decoded


# Trainer
# ---------------------------------------------------------------------------- #
class Trainer:
    """
    Class to perform model training in the style of PyTorch Lightning.
    """
    def __init__(self, model, train_generator, val_generator, device, criterion,
                 optimizer, epoch_scheduler, batch_scheduler, patience, epochs,
                 checkpoints_path) -> None:
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The neural network model to be trained.
            train_generator (DataLoader): The data loader for training data.
            val_generator (DataLoader): The data loader for validation data.
            device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            epoch_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler based on epochs.
            batch_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler based on batches.
            patience (int): The number of epochs to wait for improvement before early stopping.
            epochs (int): The maximum number of epochs for training.
            checkpoints_path (str): The directory path to save model checkpoints.
        """
        self.model = model
        self.best_model = copy.deepcopy(self.model)
        self.train_dataloader = train_generator
        self.val_dataloader = val_generator
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.batch_scheduler = batch_scheduler
        self.max_epochs = epochs
        self.best_val_loss = float('inf')
        self.patience = patience
        self.current_patience = 0
        self.checkpoints_path = checkpoints_path

    # Training #
    def __on_train_start(self, verbose=False):
        """
        Callback function called at the start of training.

        Args:
            verbose (bool): Whether to print verbose information.
        """
        if verbose:
            print("Training is started!")


    def __on_train_epoch_start(self, epoch, verbose=False):
        """
        Callback function called at the start of each training epoch.

        Args:
            epoch (int): The current epoch number.
            verbose (bool): Whether to print verbose information.
        """
        # if verbose:
        #     print(f"\nCurrent Epoch [{epoch+1}/{self.max_epochs}]")
        self.model.train(True)


    def __on_train_batch_start(self, batch_x, batch_y):
        """
        Callback function called at the start of each training batch.

        Args:
            batch_x (torch.Tensor): The input batch data.
            batch_y (torch.Tensor): The target batch data.

        Returns:
            float: The loss value for the batch.
        """
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        outputs = self.model(batch_x)

        if len(outputs.shape) < len(batch_y.shape):
            outputs = outputs.unsqueeze(1)

        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()

        if self.batch_scheduler is not None:
            self.batch_scheduler.step()

        return loss.item()


    # Validation #
    def __on_val_start(self):
        """
        Callback function called at the start of validation.
        """
        self.model.train(False)
        self.model.eval()


    def __on_val_batch_start(self, batch_x_val, batch_y_val):
        """
        Callback function called at the start of each validation batch.

        Args:
            batch_x_val (torch.Tensor): The input batch data for validation.
            batch_y_val (torch.Tensor): The target batch data for validation.

        Returns:
            float: The validation loss value for the batch.
        """
        batch_x_val = batch_x_val.to(self.device, dtype=torch.float32)
        batch_y_val = batch_y_val.to(self.device, dtype=torch.float32)

        val_outputs = self.model(batch_x_val)
        if len(val_outputs.shape) < len(batch_y_val.shape):
            val_outputs = val_outputs.unsqueeze(1)

        val_loss = self.criterion(val_outputs, batch_y_val).item()

        return val_loss


    def __early_stoping(self, val_loss, train_loss, epoch, verbose=False):
        """
        Performs early stopping based on the validation loss.

        Args:
            val_loss (float): The validation loss value.
            train_loss (float): The training loss value.
            epoch (int): The current epoch number.
            verbose (bool, optional): If True, prints verbose information. Defaults to False.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.train_loss_when_best_val_loss = train_loss
            self.current_patience = 0
            self.best_model = copy.deepcopy(self.model)

            if exist_dir(os.path.dirname(self.checkpoints_path)):
                torch.save(
                    self.model.state_dict(),
                    f'{self.checkpoints_path}'
                )

            # if verbose:
                # print(f'\t{Colors.VERDE}» Best model saved at Epoch [{epoch+1}/{self.max_epochs}]{Colors.RESET}')
        else:
            self.current_patience += 1

        if self.current_patience >= self.patience:
            if verbose:
                print(f'{Colors.ROJO}Training stopped as validation loss did not improve for {Colors.AMARILLO}{self.patience}{Colors.ROJO} epochs.{Colors.RESET}')
            return True
        return False


    # Fit #
    def fit(self, verbose=False):
        """
        Performs model training while monitoring validation accuracy and enabling LR schedulers.

        Args:
            verbose (bool, optional): If True, prints verbose information. Defaults to False.

        Returns:
            tuple: Trained model, training loss, validation loss, training loss at the best validation loss, 
                   and best validation loss.
        """
        self.train_loss_when_best_val_loss = np.inf
        self.best_val_loss = np.inf
        train_losses = []
        val_losses = []
        self.__on_train_start(verbose=verbose)

        epochs_progress_bar = tqdm(
            range(self.max_epochs),
            total=self.max_epochs,
            disable=not verbose,
            leave=True,
            unit='epoch',
            desc='Epochs loop'
        )

        for epoch in epochs_progress_bar:
            # Train
            # __________________________________________________________________
            train_loss = 0.0
            self.__on_train_epoch_start(epoch, verbose=verbose)

            # Iterate over batches
            batches_progress_bar = tqdm(
                self.train_dataloader,
                disable=not verbose,
                leave=False,
                unit='batch',
                desc='Batches loop',
            )
            for batch_x, batch_y in batches_progress_bar:
                train_loss += self.__on_train_batch_start(
                    batch_x,
                    batch_y
                ) * len(batch_x)

                if verbose:
                    train_loss_str = train_loss / len(self.train_dataloader.dataset)
                    batches_progress_bar.set_postfix_str(
                        f"Train loss: {train_loss_str:.4f} - Val loss: {val_losses[-1]:.4f}" 
                        if val_losses else f"Train loss: {train_loss_str:.4f} - Val loss: N/A"
                    )

            # Calculate average loss
            train_loss /= len(self.train_dataloader.dataset)
            train_losses.append(train_loss)

            if verbose:
                #print(f'\t» Train Loss: {train_loss}')
                epochs_progress_bar.set_postfix_str(
                    f"Train loss: {train_losses[-1]:.4f} - Val loss: {val_losses[-1]:.4f}" 
                    if train_losses and val_losses else 
                    f"Train loss: {train_losses[-1]:.4f}" if train_losses else 
                    "Train loss: N/A - Val loss: N/A"
                )

            # Validation
            # __________________________________________________________________
            if self.val_dataloader is not None:
                val_loss = 0.0
                self.__on_val_start()
                # Disable gradient calculation
                with torch.no_grad():
                    # Iterate over validation batches
                    for batch_x_val, batch_y_val in self.val_dataloader:
                        val_loss += self.__on_val_batch_start(
                            batch_x_val,
                            batch_y_val
                        ) * len(batch_x_val)

                # Calculate average validation loss
                val_loss /= len(self.val_dataloader.dataset)
                val_losses.append(val_loss)

                if verbose:
                    # print(f'\t» Val Loss: {val_loss}')
                    epochs_progress_bar.set_postfix_str(
                        f"Train loss: {train_losses[-1]:.4f}" if train_losses else "Train loss: N/A" + 
                        f" - Val loss: {val_losses[-1]:.4f}" if val_losses else " - Val loss: N/A"
                    )

            # Early Stopping
            val_loss, train_loss = (val_loss, train_loss) if self.val_dataloader is not None else (train_loss, train_loss)
            if self.__early_stoping(val_loss, train_loss, epoch, verbose=verbose):
                break

            # LR Scheduler
            if self.epoch_scheduler:
                try:
                    self.epoch_scheduler.step()
                except Exception as e:
                    self.epoch_scheduler.step(val_loss)
                except:
                    pass


        return self.best_model, train_losses, val_losses, self.train_loss_when_best_val_loss, self.best_val_loss


    def eval_dataloader(self, data_generator):
        """
        Performs evaluation on the input data loaded via a DataLoader.

        Args:
            data_generator (DataLoader): Batch generator.

        Returns:
            list: List containing predictions for the batches as torch tensors.
        """
        predictions = []
        self.best_model.eval()
        for batch_x, _ in data_generator:
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            predictions.append(self.best_model(batch_x))

        return predictions


class XAI_benchmark:
    """
    Class to perform model training in the style of PyTorch Lightning.
    """
    def __init__(self, is_ts:bool = False, model_params:dict = None, verbose:bool = True) -> None:
        self.is_ts = is_ts
        self.verbose = verbose

        if model_params['ridge']:
            self.ridge = Ridge(**model_params['ridge'])
        else:
            self.ridge = None
        if model_params['pls']:
            self.pls = PLSRegression(**model_params['pls'])
        else:
            self.pls = None
        if model_params['tree']:
            self.decision_tree = DecisionTreeRegressor(**model_params['tree'])
        else:
            self.decision_tree = None
        if model_params['svm']:
            self.svr = SVR(**model_params['svm'])
        else:
            self.svr = None
        if model_params['knn']:
            self.knn = KNeighborsRegressor(**model_params['knn'])
        else:
            self.knn = None
        if self.is_ts:
            if model_params['auto_arima']:
                self.auto_arima = auto_arima(**model_params['auto_arima'])
                print(self.auto_arima.summary())
            else:
                self.auto_arima = None
            if model_params['arima']:
                self.arima = ARIMA(**model_params['arima'])
            else:
                self.arima = None


    def fit(self, X:np.array, y:np.array) -> None:
        """
        Fit all XAI models

        Args:
            X (np.array): input training data.
            y (np.array): target training data.
        """
        if self.verbose:
            print('Fitting Ridge model...')
        if self.ridge:
            self.ridge.fit(X, y)
        if self.verbose:
            print('Fitting Partial Least Squares model...')
        if self.pls:
            self.pls.fit(X, y)
        if self.verbose:
            print('Fitting Decision Tree model...')
        if self.decision_tree:
            self.decision_tree.fit(X, y)
        if self.verbose:
            print('Fitting Support Vector Machine model...')
        if self.svr:
            self.svr.fit(X, y)
        if self.verbose:
            print('Fitting K-Nearest Neighbours model...')
        if self.knn:
            self.knn.fit(X, y)
        if self.is_ts:
            model = ''
            if self.arima:
                self.arima.fit(y)
                model = 'ARIMA'
            elif self.auto_arima:
                # self.arima = self.auto_arima.fit(y)
                model = 'Auto ARIMA'
            if self.verbose:
                print(f'Fitting {model} model...')

        if self.verbose:
            print('All models fitted!')


    def predict(self, X:np.array, y_true:np.array = None, n_periods:int = None, get_metrics:bool = False) -> dict:
        """
        Predict with all XAI models.

        Args:
            X (np.array): input validation/test data.

        Returns:
            dictionary: dictionary with all the models predictions.
        """
        predictions = {
            'ridge': self.ridge.predict(X) if self.ridge else None,
            'pls': self.pls.predict(X) if self.pls else None,
            'decision_tree': self.decision_tree.predict(X) if self.decision_tree else None,
            'svm': self.svr.predict(X) if self.svr else None,
            'knn': self.knn.predict(X) if self.knn else None,
            'arima': self.arima.predict(n_periods=n_periods, X=X) if n_periods and self.is_ts and self.arima else None,
            'auto_arima': self.auto_arima.predict(n_periods=n_periods) if n_periods and self.is_ts and self.auto_arima else None
        }
        metrics = {}

        if get_metrics:
            assert y_true is not None, 'y must be provided to calculate metrics.'
            for model in ['ridge', 'pls', 'decision_tree', 'svm', 'knn', 'arima', 'auto_arima']:
                if predictions.get(model) is not None:
                    metrics[model] = {
                        'mse': mean_squared_error(y_true, predictions[model]),
                        'rmse': np.sqrt(mean_squared_error(y_true, predictions[model])),
                        'mae': mean_absolute_error(y_true, predictions[model]),
                        'mape': mean_absolute_percentage_error(y_true, predictions[model]),
                        'R2': r2_score(y_true, predictions[model])
                    }
                else:
                    metrics[model] = {
                        'mse': None,
                        'rmse': None,
                        'mae': None,
                        'mape': None,
                        'R2': None
                    }

        return predictions, metrics


    def save(self, path:str, subfix:str = ''):
        """
        Save the XAI models in pickle format.

        Args:
            path (str): path to save the models.
            subfix (str): extra name for info.
        """
        names = ['ridge', 'pls', 'decision_tree', 'svr', 'knn']
        models = [self.ridge, self.pls, self.decision_tree, self.svr, self.knn]
        for name, model in zip(names, models):
            file_name = f'{name}_{subfix}.pkl' if subfix else f'{name}.pkl'
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(model, f)

        file_name = f'arima_{subfix}.pkl' if subfix else 'arima.pkl'
        if self.is_ts:
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(self.arima, f)


    def load(self, folder_path:str):
        """
        Save the XAI models in pickle format.

        Args:
            folder_path (str): path to saved models.
        """
        names = ['ridge', 'pls', 'decision', 'svr', 'knn', 'arima']
        models = [self.ridge, self.pls, self.decision_tree, self.svr, self.knn, self.arima]
        loaded_models = {}

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path):
                subfix = file_name.split('_')[0]
                try:
                    i_list = names.index(subfix)
                    with open(full_path, 'rb') as f:
                        loaded_models[names[i_list]] = pickle.load(f)
                        print(f'Loaded {names[i_list]} model')
                except Exception as e:
                    print(f'Error loading model {file_name}: {e}')

        # Update attributes in self
        self.ridge = loaded_models.get('ridge', self.ridge)
        self.pls = loaded_models.get('pls', self.pls)
        self.decision_tree = loaded_models.get('decision', self.decision_tree)
        self.svr = loaded_models.get('svr', self.svr)
        self.knn = loaded_models.get('knn', self.knn)
        self.arima = loaded_models.get('arima', self.arima)
