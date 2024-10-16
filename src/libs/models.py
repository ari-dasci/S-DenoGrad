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
import sys

# Libraries
# ---------------------------------------------------------------------------- #
import copy
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
# Locals
from config import Colors
from utils import *

# CURRENT_DIR = os.getcwd()
# FOLDERS = CURRENT_DIR.split(os.sep)
# TESIS_FOLDER_INDEX = FOLDERS.index('Tesis')
# CURRENT_DIR = os.sep.join(FOLDERS[:TESIS_FOLDER_INDEX+1])
# CURRENT_DIR = os.path.join(CURRENT_DIR, 'S-noise-gradient')
# LIBS_PATH = os.path.join(CURRENT_DIR, 'src', 'libs')
# assert os.path.exists(LIBS_PATH)
# sys.path.append(LIBS_PATH)


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
        # try:
        #     outputs = self.model(batch_x).squeeze()
        # except:
        #     outputs = self.model(batch_x.view(batch_x.shape[0], 1))
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
        # try:
        #     val_outputs = self.model(batch_x_val).squeeze()
        # except:
        #     val_outputs = self.model(batch_x_val.view(batch_x_val.shape[0], 1))
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
                    f'{self.checkpoints_path}.pth'
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
                    batches_progress_bar.set_postfix_str(f"Train loss: {train_loss / len(self.train_dataloader.dataset)} - Val loss: {val_losses[-1] if val_losses else 'N/A'}")

            # Calculate average loss
            train_loss /= len(self.train_dataloader.dataset)
            train_losses.append(train_loss)

            if verbose:
                #print(f'\t» Train Loss: {train_loss}')
                epochs_progress_bar.set_postfix_str(f"Train loss: {train_losses[-1] if train_losses else 'N/A'} - Val loss: {val_losses[-1] if val_losses else 'N/A'}")

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
                    epochs_progress_bar.set_postfix_str(f"Train loss: {train_losses[-1] if train_losses else 'N/A'} - Val loss: {val_losses[-1] if val_losses else 'N/A'}")

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
