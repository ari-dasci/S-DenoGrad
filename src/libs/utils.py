# pylint: disable=invalid-name
"""
This module contains some useful functions and classes to perform
different tasks such as data visualization, data preprocessing and
others. It also contains some general functions that are used in
other parts of the project.

Functions:
    na_heatmap: Show a heatmap about the missing data of df
    plot_selected_variable: Plot the time series variable by variable using widget
    get_correlated: Get the names of variables correlated with "variable" that have
                    a correlation value greater than "threshold"
    get_outliers: Get outliers in variable
    win_generator: Sliding window generator 'on-the-fly'
    prep_forcasted_data_to_inverse_scale: Adjusts the data vector resulting from the LSTM inference
                                          to a format that can be used by the scaler so that it can
                                          perform the inverse scaling of the data
    split_dataset: Split the dataset into training and validation sets
    win_data_generator: Create an on-the-fly sliding window generator that returns
                       window_size values of X and their corresponding labels from Y
    plot_predictions: Plot predictions along with the original data or the
                      evolution of loss during training both in training and validation
    add_gaussian_noise: Adds Gaussian noise to specified columns in a DataFrame
    exist_dir: Checks if a directory exists and, if not, creates it

Classes:
    MinMaxScalerCustom: Min Max Scaler with custom range for each column
"""

# Add libs path to sys path and create some global path variables
# ---------------------------------------------------------------------------- #
import os
import sys

# Import libraries
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import widgets
from torch.utils.data import DataLoader
# Locals
from dataset import SlidingWindowDataset


# EDA Functions
# ---------------------------------------------------------------------------- #
def na_heatmap(df, cmap='viridis', label='Missing Data Heatmap'):
    """
    Show a heatmap about the missing data of df

    Args:
        df (pd.DataFrame): DataFrame to analyze
        label (str, optional): Title of the graph. Defaults to 'Missing Data Heatmap'.
    """
    plt.figure(figsize=(10, 15))
    sns.heatmap(df.isnull(), cmap=cmap, cbar=False, yticklabels=False)
    plt.title(label)
    plt.show()


def plot_selected_variable(df, variable_name):
    """
    Function to plot the time series variable by variable using widget

    Args:
        dataframes_dic (dictionary): dictionary with all dataframes
        dataframe_name (str): name of the dataframe to display
        variable_name (str): name of the variable to display
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[variable_name], label=variable_name, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Multivariable Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_correlated(corr_matrix, variable, threshold):
    """
    Get the names of variables correlated with "variable" that have
    a correlation value greater than "threshold"

    Args:
        corr_matrix (DataFrame): correlation matrix
        variable (str): name of the variable from which we want to find
                correlations
        threshold (float): threshold for the correlation value

    Returns:
        list: list of str with the names of correlated variables
    """
    corr_variables = []
    for i, value in enumerate(corr_matrix[variable]):
        if abs(value) > threshold:
            var_name = corr_matrix.index[i]
            if var_name != variable:
                corr_variables.append(var_name)

    return corr_variables


def get_outliers(df, variable):
    """
    Get outliers in variable

    Args:
        df (DataFrame): dataset
        variable (str): name of the dataframe column

    Returns:
        dataframe: boolean column indicating outliers data
    """
    # Calculate the IQR
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    LI = Q1 - 1.5 * IQR
    LS = Q3 + 1.5 * IQR

    # Identify and count outlier values
    outliers = df[(df[variable] < LI) | (df[variable] > LS)]

    return outliers


# General functions
# ---------------------------------------------------------------------------- #
def win_generator(df, target_column, timesteps, future):
    """
    Sliding window generator 'on-the-fly'

    Args:
        df (pd.DataFrame): dataset
        target_column (str): target column
        timesteps (int): window size
        future (int): future instant to predict. 0 is the instant
                immediately after the sliding window.

    Yields:
        tuple: sliding window and target value
    """
    for i in range(df.shape[0] - timesteps - future):
        seq = df.iloc[i:i + timesteps].values
        label = df.iloc[i + timesteps + future, target_column]
        yield seq, label


def prep_forcasted_data_to_inverse_scale(forcasted_data, df_cols, data_column,
                                         full_of=np.nan):
    """
    Adjusts the data vector resulting from the LSTM inference to a
    format that can be used by the scaler so that it can perform
    the inverse scaling of the data.

    Args:
        forcasted_data (np.array): prediction result data
        df_cols (list(str)): list with the names of the dataset columns
                with which it has been inferred
        data_column ((str,int)): index or name of the column occupying the
                inferred variable

    Raises:
        TypeError: In case the column is indicated with an incorrect data type

    Returns:
        pd.DataFrame: dataframe filled with np.nan except for the column that
                we want to perform inverse scaling
    """
    n_rows = len(forcasted_data)
    n_cols = len(df_cols)
    matrix = np.full((n_rows, n_cols), full_of)
    _df = pd.DataFrame(matrix, columns=df_cols)

    col = None
    if isinstance(data_column, int):
        col = df_cols[data_column]
    elif isinstance(data_column, str):
        col = data_column
    else:
        raise TypeError("Column is expected to be indicated with integer \
                        index or its name (str)")

    _df[col] = forcasted_data

    return _df


def split_dataset(df, ratio):
    """
    Split the dataset into training and validation sets

    Args:
        df (pd.DataFrame): complete dataset
        ratio (float): percentage dedicated to the training set

    Returns:
        tuple: training and validation sets
    """
    assert 1 > ratio > 0
    last_item = round(df.shape[0]*ratio)
    return df.iloc[:last_item], df.iloc[last_item:]


def win_data_generator(X, Y, window_size, batch_size, future, shuffle=True,
                       mode='range', cnn=False):
    """
    Create an on-the-fly sliding window generator that returns
    window_size values of X and their corresponding labels from Y.

    Args:
        X (np.array): dataset
        Y (np.array): labels set
        window_size (int): sliding window size
        batch_size (int): batch size

    Yields:
        tuple: pair of np.array with data window and their corresponding labels
    """
    dataset = SlidingWindowDataset(X, Y, window_size, future, mode, cnn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def plot_predictions(variable, variables_dict, visualize, futures,
                     label1=widgets.fixed('Ground Truth '),
                     label2=widgets.fixed('Prediction '),
                     abs_label=widgets.fixed('Time'),
                     ord_label=widgets.fixed('Value'),
                     losses=widgets.fixed(False),
                     alpha=0.3
                     ):
    """
    Function to plot predictions along with the original data or the
    evolution of loss during training both in training and validation.

    Args:
        variable (str): variable to show in the graph
        variables_dict (dict): dictionary that relates variables with their
                values
        visualize (str): allows choosing what to show in the graph
        label1 (str, optional): label describing the first data to
                show. Defaults to widgets.fixed('Ground Truth ').
        label2 (str, optional): label describing the second data to
                show. Defaults to widgets.fixed('Prediction ').
        abs_label (str, optional): label of the abscissa axis. Defaults to
                widgets.fixed('Time').
        ord_label (str, optional): label of the ordinate axis. Defaults to
                widgets.fixed('Value').
    """
    x, x_ = variables_dict[variable]
    if not losses:
        x_ = x_[futures]
    plt.figure(figsize=(15, 10))
    if visualize in ["First", "Both"]:
        plt.plot(
            x,
            label=label1 + variable,
            color='blue',
            alpha=alpha,
            linestyle='--',
            marker='x'
        )
    if visualize in ["Second", "Both"]:
        plt.plot(
            x_,
            label=label2 + variable,
            color='red',
            alpha=alpha,
            linestyle='-',
            marker='o'
        )
    plt.xlabel(abs_label)
    plt.ylabel(ord_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def add_gaussian_noise(df, columns, mean=0, std=0.1):
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


def exist_dir(dir_path):
    """
    Checks if a directory exists and, if not, creates it.

    Args:
        dir_path (str): path of the directory to check

    Returns:
        bool: True if the directory already exists or has been created successfully
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True


# Classes
# ---------------------------------------------------------------------------- #
class MinMaxScalerCustom:
    """
    Min Max Scaler with custom range for each column
    """
    def __init__(self, min_max_dict):
        self.min_max_dict = min_max_dict


    def scale(self, df, column):
        """
        Scale the data of a dataframe between [0,1]

        Args:
            df (pd.Dataframe): Dataframe to scale
            column (str): column name to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        min_val = self.min_max_dict[column][0]
        max_val = self.min_max_dict[column][1]
        df[column] = (df[column] - min_val) / (max_val - min_val)
        return df


    def scale_value(self, value, column):
        """
        Scale the data of a dataframe between [0,1]

        Args:
            df (pd.Dataframe): Dataframe to scale
            column (str): column name to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        min_val = self.min_max_dict[column][0]
        max_val = self.min_max_dict[column][1]
        value = (value - min_val) / (max_val - min_val)
        return value


    def transform(self, df, values=True):
        """
        Scale the data of a dataframe between [0,1]

        Args:
            df (pd.Dataframe): Dataframe to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        for column in df.columns:
            df = self.scale(df, column)

        if values:
            return df.values
        return df


    def unscale(self, df, column):
        """
        Return the data to the original scale

        Args:
            df (pd.Dataframe): Dataframe to scale
            column (str): column name to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        min_val = self.min_max_dict[column][0]
        max_val = self.min_max_dict[column][1]
        df[column] = df[column] * (max_val - min_val) + min_val
        return df


    def unscale_value(self, value, column):
        """
        Return the data to the original scale

        Args:
            df (pd.Dataframe): Dataframe to scale
            column (str): column name to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        min_val = self.min_max_dict[column][0]
        max_val = self.min_max_dict[column][1]
        value = value * (max_val - min_val) + min_val
        return value


    def inverse_transform(self, df, values=True):
        """
        Return the data to the original scale

        Args:
            df (pd.Dataframe): Dataframe to scale

        Returns:
            pd.Dataframe: scaled dataframe
        """
        for column in df.columns:
            df = self.unscale(df, column)

        if values:
            return df.values
        return df
