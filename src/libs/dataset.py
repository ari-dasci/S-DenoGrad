"""
This module contains the classes to structure the data as a sliding window and
the custom dataset for the symbolic regression problem on BioFuel.

Classes:
    - SlidingWindowDataset: Class to structure data as a sliding window
    - BioFuelDataset: Custom dataset for symbolic regression problem on BioFuel
"""

# Libraries
# ---------------------------------------------------------------------------- #
import copy
import numpy as np
import torch
from torch.utils.data import Dataset

# Classes
# ---------------------------------------------------------------------------- #
class SlidingWindowDataset(Dataset):
    """
    Class to structure data as a sliding window

    Args:
        Dataset (Class): Inherits from the Dataset class
    """
    def __init__(self, X, Y, window_size, future, mode='range', cnn=False):
        """
        Initialize the sliding window dataset

        Args:
            X (pd.DataFrame): input data set
            Y (pd.DataFrame): label data set
            window_size (int): size of the sliding window
            future (int|list(int)): maximum future to predict | list of 
                    futures to predict
            mode (str, optional): range or discrete. Defaults to 'range'.
                    Mode in which the future is interpreted.
                    - range: the future is of type int; all instants up to it are predicted.
                    - discrete: the future is of type list(int);
                        the indicated futures are predicted.
            cnn (bool, optional): indicates if the LSTM model to be used
                    has convolutional layers in the input. Defaults to False.
        """
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.future = future
        self.mode = mode
        self.is_cnn = cnn


    def __len__(self):
        if self.mode == 'range':
            n_windows = len(self.X) - self.window_size - self.future + 1
        elif self.mode == 'discrete':
            n_windows = len(self.X) - self.window_size - max(self.future) + 1

        return n_windows


    def __getitem__(self, idx):
        assert idx < len(self), f'Index {idx} out of range'
        x = self.X.iloc[idx:idx + self.window_size].values
        if self.is_cnn:
            x = x.T

        if self.mode == 'range':
            y = self.Y.iloc[idx + self.window_size : idx + self.window_size + self.future].values
        elif self.mode == 'discrete':
            y = np.array([self.Y.iloc[idx + self.window_size + i_fut] for i_fut in self.future])

        return [x, y]


    def __iter__(self):
        """
        Devuelve un iterador sobre el dataset, para que se pueda usar en un bucle for.
        """
        for idx in range(len(self)):
            yield self.__getitem__(idx)


    def copy(self):
        """
        Returns a deep copy of the current instance of SlidingWindowDataset.
        """
        return copy.deepcopy(self)


class Dataset(Dataset):
    """
    Custom dataset class

    Args:
        Dataset (Class): Inherits from the torch.utils.data.Dataset class
    """
    def __init__(self, input_df, target_df):
        self.input_df = input_df
        self.target_df = target_df
        assert len(self.input_df) == len(self.target_df)

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.input_df.iloc[idx].tolist(), dtype=torch.float32)
        target = torch.tensor(self.target_df.iloc[idx].tolist(), dtype=torch.float32)

        return features, target
