"""
This module provides a base class `BaseExperiment` and its specialized subclasses 
`TSExperiment` and `TabularExperiment` for conducting experiments involving data 
processing, noise addition, denoising, and Explainable AI (XAI) benchmarking. 

Classes:
    BaseExperiment:
        A base class for managing the experimental pipeline, including data loading, 
        noise addition, denoising, XAI benchmarking, and metric calculation. It provides 
        methods for handling various stages of the experiment and saving results.

    TSExperiment:
        A subclass of `BaseExperiment` designed specifically for time-series data. 
        It includes methods for loading and preprocessing time-series data, scaling, 
        and splitting it into training and testing sets.

    TabularExperiment:
        A subclass of `BaseExperiment` designed for tabular data. It includes methods 
        for loading and preprocessing tabular data, scaling, and splitting it into 

Key Features:
    - Data loading and preprocessing for both time-series and tabular data.
    - Addition of Gaussian noise to datasets.
    - Application of denoising methods to noisy data.
    - Execution of XAI benchmarking on original, noisy, and denoised datasets.
    - Calculation of various metrics, including histograms, KL divergence, correlations, 
      and distances between datasets.
    - Saving of predictions and metrics to JSON files.

    is_ts (bool): Indicates whether the data is time-series (set in subclasses).

        Abstract method to load and preprocess data. Must be implemented in subclasses.

    add_noise(sigma: float = 0.02) -> None:

    perform_denoising(denoise_method: callable, **kwargs) -> None:
        Applies a specified denoising method to the noisy data.

    perform_original_xai_benchmark(model_params: dict, verbose: bool = True) -> None:
        Executes the XAI benchmark on the original dataset.

    perform_noisy_xai_benchmark(model_params: dict, verbose: bool = True) -> None:
        Executes the XAI benchmark on the noisy dataset.

    perform_denoised_xai_benchmark(model_params: dict, verbose: bool = True) -> None:
        Executes the XAI benchmark on the denoised dataset.

    calculate_metrics() -> None:
        Calculates various metrics such as histograms, KL divergence, correlations, 
        and distances between datasets.

    save_results() -> None:
        Saves predictions and metrics to JSON files.

    run(data_file: str, y_col_name: str = '', add_noise: bool = False, sigma: float = 0.02, 
        denoising_method: callable = None, denoising_method_params: dict = None, 
        xai_models_params: dict = None) -> None:

    dictionary_arrays_to_list(array_d):
        Static method to recursively convert arrays in a dictionary to lists for JSON serialization.
"""
# Standard library imports
import os
import json
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import entropy

# Local imports
from src.libs.utils import add_gaussian_noise, symmetric_mean_absolute_percentage_error
from src.libs.xai_benchmark import XAIBenchmark

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# BaseExperiment #
# ------------------------------------------------------------------------------------------------ #
class BaseExperiment:
    """
    BaseExperiment is a base class designed to facilitate experiments involving data processing, 
    noise addition, denoising, and explainable AI (XAI) benchmarking. It provides methods for 
    loading data, adding noise, applying denoising techniques, performing XAI benchmarks, and 
    calculating various metrics to evaluate the impact of noise and denoising on the data.
    Attributes:
        data_path (str): Path to the input data.
        out_path (str): Path to save the output results.
        checkpoint_path (str): Path to save or load model checkpoints.
        subfix_name (str): A suffix name used for saving results.
        is_ts (bool): Indicates whether the data is time-series (to be set in subclasses).
        is_cnn (bool): Indicates whether the model is a CNN.
        should_train_original_xai (bool): Flag to train XAI models on original data.
        should_train_noisy_xai (bool): Flag to train XAI models on noisy data.
        should_train_denoised_xai (bool): Flag to train XAI models on denoised data.
        should_train_denoising_method (bool): Flag to train a denoising method.
        original_data (dict): Dictionary to store original data and related attributes.
        noisy_data (dict): Dictionary to store noisy data and related attributes.
        denoised_data (dict): Dictionary to store denoised data and related attributes.
        predictions_dict (dict): Dictionary to store model predictions.
        metrics_dict (dict): Dictionary to store calculated metrics.
    Methods:
        load_data(): Abstract method to load and preprocess data. Must be implemented in subclasses.
        add_noise(sigma): Adds Gaussian noise to the original data.
        perform_denoising(denoise_method, **kwargs): Applies a denoising method to the noisy data.
        perform_original_xai_benchmark(model_params, verbose): Runs XAI benchmarking on original
            data.
        perform_noisy_xai_benchmark(model_params, verbose): Runs XAI benchmarking on noisy data.
        perform_denoised_xai_benchmark(model_params, verbose): Runs XAI benchmarking on denoised
            data.
        calculate_metrics(): Calculates various metrics such as histograms, KL divergence,
            correlations, and distances between datasets.
        save_results(): Saves predictions and metrics to JSON files.
        run(add_noise, sigma, denoising_method, denoising_method_params, xai_models_params): 
            Executes the full experimental pipeline, including data loading, noise addition, 
            denoising, XAI benchmarking, metric calculation, and result saving.
    Static Methods:
        dictionary_arrays_to_list(array_d): Converts arrays in a dictionary to lists for JSON
            serialization.
    """
    def __init__(self, data_path: str, out_path: str, checkpoint_path: str, subfix_name: str,
                 is_cnn: bool = False, train_original_xai: bool = False,
                 train_noisy_xai: bool = False, train_denoised_xai: bool = False,
                 train_denoising_method: bool = False, verbose: bool = True) -> None:

        # Paths and names
        self.data_path = data_path
        self.out_path = out_path
        self.checkpoint_path = checkpoint_path
        self.subfix_name = subfix_name

        # Boolean flags
        self.is_ts = None
        self.is_cnn = is_cnn
        self.verbose = verbose
        self.should_train_original_xai = train_original_xai
        self.should_train_noisy_xai = train_noisy_xai
        self.should_train_denoised_xai = train_denoised_xai
        self.should_train_denoising_method = train_denoising_method

        # Data
        self.original_data = {}
        self.noisy_data = {}
        self.denoised_data = {}

        # Model predictions and metrics
        self.predictions_dict = {}
        self.metrics_dict = {}

    def load_data(self, data_file: str, y_col_name: str) -> tuple:
        """
        Load data from a specified file and extract the target column.
        This method is intended to be implemented in a subclass to handle
        the loading and processing of data specific to the experiment.
        Args:
            data_file (str): The path to the data file to be loaded.
            y_col_name (str): The name of the column containing the target variable.
        Returns:
            tuple: A tuple containing the features and target variable.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in the subclass.")

    def add_noise(self, sigma: float = 0.02) -> None:
        """
        Adds Gaussian noise to the original data and stores the result in the noisy_data attribute.
        Args:
            sigma (float, optional): The standard deviation of the Gaussian noise to be added. 
                Defaults to 0.02. The value is rounded to three decimal places.
        Returns:
            None
        """
        sigma = round(sigma, 3)
        self.noisy_data['df'] = add_gaussian_noise(
            df=self.original_data['df'].copy(),
            columns=list(self.original_data['df'].columns),
            mean=0.0,
            std=sigma
        )

    def _perform_xai_benchmark(self, model_params: dict, train_data_dict: dict,
                               test_data_dict: dict, checkpoint_folder: str, should_train: bool,
                               subfix: str) -> None:
        """
        Perform the XAI (Explainable AI) benchmark for a given model and dataset.
        This method evaluates the performance of a model using an XAI benchmark, 
        either by training the benchmark model on the provided data or by loading 
        a pre-trained model from a checkpoint. The results, including predictions 
        and metrics, are stored in the corresponding dictionaries.
        Args:
            model_params (dict): Parameters for the model to be used in the XAI benchmark.
            data_dict (dict): Dictionary containing the training and testing data. 
                                Expected keys are 'x_train', 'y_train', 'x_test', and 'y_test'.
            checkpoint_folder (str): Path to the folder where the checkpoint files are stored or
                will be saved.
            should_train (bool): Flag indicating whether to train the model on the provided data.
            subfix (str): A suffix to identify the specific benchmark run in the saved files and
                metrics.
        Returns:
            None
        """
        xai_benchmark = XAIBenchmark(
            is_ts = self.is_ts,
            model_params = model_params,
            verbose = self.verbose
        )

        if should_train:
            xai_benchmark.fit(train_data_dict['x_train'], train_data_dict['y_train'])
        else:
            xai_benchmark.load(os.path.join(self.checkpoint_path, 'xai', checkpoint_folder))

        predictions, metrics = xai_benchmark.predict(
            test_data_dict['x_test'],
            test_data_dict['y_test'],
            n_periods=len(test_data_dict['y_test']),
            get_metrics=True
        )

        xai_benchmark.save(
            path = os.path.join(self.checkpoint_path, 'xai', checkpoint_folder)
        )

        if 'XAI' not in self.metrics_dict:
            self.metrics_dict['XAI'] = {}
            self.predictions_dict['XAI'] = {}

        if subfix not in self.metrics_dict['XAI']:
            self.metrics_dict['XAI'][subfix] = {}
            self.predictions_dict['XAI'][subfix] = {}

        aux1 = f"train_{train_data_dict['df'].name}_test_{test_data_dict['df'].name}"
        aux2 = f"train_{train_data_dict['df'].name}_test_{test_data_dict['df'].name}"

        self.metrics_dict['XAI'][subfix][aux1] = metrics
        self.predictions_dict['XAI'][subfix][aux2] = predictions

    def perform_original_xai_benchmark(self, model_params: dict) -> None:
        """
        Executes the original Explainable AI (XAI) benchmark for the given model parameters.
        This method performs the XAI benchmark using the original dataset and saves the results
        in a designated checkpoint folder with a specific subfix for identification.
        Args:
            model_params (dict): A dictionary containing the parameters for the model
                to be evaluated.
        Returns:
            None
        """
        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.original_data,
            test_data_dict=self.original_data,
            checkpoint_folder='orig',
            should_train=self.should_train_original_xai,
            subfix='orig'
        )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.original_data,
            test_data_dict=self.noisy_data,
            checkpoint_folder='orig',
            should_train=self.should_train_original_xai,
            subfix='orig'
        )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.original_data,
            test_data_dict=self.denoised_data,
            checkpoint_folder='orig',
            should_train=self.should_train_original_xai,
            subfix='orig'
        )

    def perform_noisy_xai_benchmark(self, model_params: dict) -> None:
        """
        Executes the XAI (Explainable Artificial Intelligence) benchmark on noisy data.
        This method performs an XAI benchmark using the provided model parameters and 
        the noisy dataset. It saves the results in a designated checkpoint folder and 
        appends a specific subfix to the output files.
        Args:
            model_params (dict): A dictionary containing the parameters for the model 
                to be used in the benchmark.
        Returns:
            None
        """
        if 'df' in self.original_data:
            if self.original_data['df'] is not None:
                self._perform_xai_benchmark(
                    model_params=model_params,
                    train_data_dict=self.noisy_data,
                    test_data_dict=self.original_data,
                    checkpoint_folder='noisy',
                    should_train=self.should_train_noisy_xai,
                    subfix='noisy'
                )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.noisy_data,
            test_data_dict=self.noisy_data,
            checkpoint_folder='noisy',
            should_train=self.should_train_noisy_xai,
            subfix='noisy'
        )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.noisy_data,
            test_data_dict=self.denoised_data,
            checkpoint_folder='noisy',
            should_train=self.should_train_noisy_xai,
            subfix='noisy'
        )

    def perform_denoised_xai_benchmark(self, model_params: dict) -> None:
        """
        Executes the XAI (Explainable Artificial Intelligence) benchmark on denoised data.
        This method performs an XAI benchmark using the provided model parameters and 
        the denoised dataset. The results are saved in a specific checkpoint folder 
        with a designated subfix for identification.
        Args:
            model_params (dict): A dictionary containing the parameters for the model 
                to be used in the benchmark.
        Returns:
            None
        """
        if 'df' in self.original_data:
            if self.original_data['df'] is not None:
                self._perform_xai_benchmark(
                    model_params=model_params,
                    train_data_dict=self.denoised_data,
                    test_data_dict=self.original_data,
                    checkpoint_folder='denoised',
                    should_train=self.should_train_denoised_xai,
                    subfix='denoised'
                )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.denoised_data,
            test_data_dict=self.noisy_data,
            checkpoint_folder='denoised',
            should_train=self.should_train_denoised_xai,
            subfix='denoised'
        )

        self._perform_xai_benchmark(
            model_params=model_params,
            train_data_dict=self.denoised_data,
            test_data_dict=self.denoised_data,
            checkpoint_folder='denoised',
            should_train=self.should_train_denoised_xai,
            subfix='denoised'
        )

    def perform_denoising(self, denoise_method: callable, **kwargs):
        """
        Applies the specified denoising method to the noisy data.

        Args:
            denoise_method (callable): A function or callable object that performs denoising.
                    It should accept the noisy data as input and return the denoised data.
            **kwargs: Additional arguments to pass to the denoising method.
        """
        if not callable(denoise_method):
            raise ValueError("The denoise_method must be a callable function or object.")

        self.metrics_dict['denoising'] = {}
        self.metrics_dict['denoising']['fitting'] = None

        # Apply the denoising method to the noisy data
        denoising_result = denoise_method(self.noisy_data, **kwargs)
        if isinstance(denoising_result, tuple) and len(denoising_result) == 2:
            denoised_data, possible_metrics = denoising_result
            self.denoised_data['df'] = denoised_data
            self.metrics_dict['denoising']['fitting'] = possible_metrics
        else:
            self.denoised_data['df'] = denoising_result

        self.denoised_data['df'] = pd.DataFrame(
            self.denoised_data['df'],
            columns=self.noisy_data['df'].columns
        )
        self.denoised_data['df'].name = 'denoised'

        if self.is_ts:
            # Shift the target variable to create a new column
            self.denoised_data['df']['y_shifted'] = self.denoised_data['df']['y'].shift(-1)
            # Drop the last row with NaN value
            self.denoised_data['df'] = self.denoised_data['df'].dropna().reset_index(drop=True)
            y_shifted = self.denoised_data['df']['y_shifted'].copy()
            self.denoised_data['df'] = self.denoised_data['df'].drop(columns=['y_shifted'])

            # divide the data into train/test datasets
            input_vars = self.denoised_data['df'].columns
            x_train, x_test, y_train, y_test = train_test_split(
                self.denoised_data['df'][input_vars].values, y_shifted, test_size=0.2, shuffle=False
            )
        else:
            # divide the data into train/test datasets
            input_vars = list(set(self.denoised_data['df'].columns) - set(['y']))
            x_train, x_test, y_train, y_test = train_test_split(
                self.denoised_data['df'][input_vars].values,
                self.denoised_data['df']['y'].values,
                test_size=0.2,
                random_state=42
            )

        self.denoised_data['x_train'] = x_train
        self.denoised_data['x_test'] = x_test
        self.denoised_data['y_train'] = y_train
        self.denoised_data['y_test'] = y_test

        gt_values = self.noisy_data['df'].values
        predicted_values = self.denoised_data['df'].values
        mae = mean_absolute_error(gt_values, predicted_values)
        smape = symmetric_mean_absolute_percentage_error(gt_values, predicted_values)
        mse = mean_squared_error(gt_values, predicted_values)
        rmse = np.sqrt(mse)
        r_squared = r2_score(gt_values, predicted_values)

        data_comparison_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'smape': smape,
            'R2': r_squared
        }

        self.metrics_dict['denoising']['datasets_comparison'] = data_comparison_metrics

    def _calc_histograms(self) -> None:
        """
        Calculate histograms for the original, noisy, and denoised data.
        This method computes histograms for each column in the dataframes stored 
        in `self.original_data`, `self.noisy_data`, and `self.denoised_data`. 
        The histograms are stored in the respective `histogram` dictionary 
        within each data dictionary. The number of bins for each column is 
        determined dynamically or reused if previously calculated.
        The histograms are normalized to represent probability density functions 
        and a small constant (1e-10) is added to avoid zero values.
        Attributes:
            self.original_data (dict): Dictionary containing the original data 
                and its associated histogram.
            self.noisy_data (dict): Dictionary containing the noisy data and 
                its associated histogram.
            self.denoised_data (dict): Dictionary containing the denoised data 
                and its associated histogram.
        Notes:
            - The method assumes that each data dictionary contains a key `'df'` 
              with a pandas DataFrame as its value.
            - If a column's histogram bins have been calculated previously, 
              the same number of bins will be reused for consistency.
        """
        self.original_data['histogram'] = {}
        self.noisy_data['histogram'] = {}
        self.denoised_data['histogram'] = {}
        histogram_bins = {}
        # Calculate histograms for each data type
        for data_dict in [self.original_data, self.noisy_data, self.denoised_data]:
            if 'df' in data_dict:
                if data_dict['df'] is not None:
                    for col in data_dict['df'].columns:
                        bins = 'auto'
                        if col in histogram_bins:
                            bins = histogram_bins[col]

                        # Calculate histogram
                        hist, bin_edgs = np.histogram(data_dict['df'][col], bins=bins, density=True)
                        data_dict['histogram'][col] = hist + 1e-10
                        histogram_bins[col] = len(bin_edgs) - 1

    def _calc_kullback_leibler_divergence(self) -> None:
        """
        Calculate the Kullback-Leibler (KL) divergence between different data distributions
        and store the results in the `metrics_dict`.
        This method computes the KL divergence for the following pairs of distributions:
        1. Original data vs. Noisy data (`orig_noisy`).
        2. Original data vs. Denoised data (`orig_denoised`).
        3. Noisy data vs. Denoised data (`noisy_denosied`).
        The results are stored in the `metrics_dict` under the key `KL_divergence`, with
        sub-keys for each pair of distributions.
        Note:
            - The method assumes that the histograms for the original, noisy, and denoised
              data are precomputed and stored in `self.original_data['histogram']`,
              `self.noisy_data['histogram']`, and `self.denoised_data['histogram']` respectively.
            - The histograms are expected to be dictionaries where keys are column names
              and values are the histogram data for each column.
        Raises:
            KeyError: If a column in `self.denoised_data['df']` does not exist in the
                      corresponding histogram dictionaries.
        """
        if 'KL_divergence' not in self.metrics_dict:
            self.metrics_dict['KL_divergence'] = {}
            self.metrics_dict['KL_divergence']['orig_noisy'] = {}
            self.metrics_dict['KL_divergence']['orig_denoised'] = {}
            self.metrics_dict['KL_divergence']['noisy_denosied'] = {}

        for col in self.denoised_data['df'].columns:
            if col in self.original_data['histogram']:
                # Calculate KL divergence between original and noisy data
                kl_divergence = entropy(
                    self.original_data['histogram'][col],
                    self.noisy_data['histogram'][col]
                )
                self.metrics_dict['KL_divergence']['orig_noisy'][col] = kl_divergence

                # Calculate KL divergence between denoised and noisy data
                kl_divergence = entropy(
                    self.original_data['histogram'][col],
                    self.denoised_data['histogram'][col]
                )
                self.metrics_dict['KL_divergence']['orig_noisy'][col] = kl_divergence

            # Calculate KL divergence between denoised and noisy data
            kl_divergence = entropy(
                self.noisy_data['histogram'][col],
                self.denoised_data['histogram'][col]
            )
            self.metrics_dict['KL_divergence']['noisy_denosied'][col] = kl_divergence

    def _calc_correlation(self) -> None:
        """
        Calculate the correlation matrix and mean correlation for data dictionaries.
        This method computes the correlation matrix for each data dictionary in 
        `self.original_data`, `self.noisy_data`, and `self.denoised_data` if the 
        dictionary contains a DataFrame under the key 'df'. The resulting correlation 
        matrix is stored in the dictionary under the key 'correlation'. Additionally, 
        the mean of the correlation matrix (ignoring NaN values) is calculated and 
        stored under the key 'mean_correlation'.
        Returns:
            None
        """
        for data_dict in [self.original_data, self.noisy_data, self.denoised_data]:
            if 'df' in data_dict:
                if data_dict['df'] is not None:
                    data_dict['correlation'] = data_dict['df'].corr()
                    # Calculate the mean correlation
                    mean_corr = np.nanmean(data_dict['correlation'])
                    data_dict['mean_correlation'] = mean_corr

    def _calc_correlation_difference_between_datasets(self) -> None:
        """
        Calculate and store the absolute differences between correlation metrics 
        across original, noisy, and denoised datasets.
        This method computes the following correlation differences:
        1. The absolute difference between the original and noisy dataset correlations.
        2. The absolute difference between the original and denoised dataset correlations.
        3. The absolute difference between the noisy and denoised dataset correlations.
        The results are stored in the `metrics_dict['correlations_diff']` dictionary 
        with the following keys:
        - 'original_noisy_correlation_diff': Difference between original and noisy correlations.
        - 'original_denoised_correlation_diff': Difference between original and denoised
            correlations.
        - 'noisy_denoised_correlation_diff': Difference between noisy and denoised correlations.
        If the original correlation is not available, the differences involving the 
        original dataset will be set to `None`.
        Returns:
            None
        """
        original_corr = None
        if 'mean_correlation' in self.original_data:
            original_corr = self.original_data['mean_correlation']

        noisy_corr = self.noisy_data['mean_correlation']
        denoised_corr = self.denoised_data['mean_correlation']

        self.metrics_dict['correlations_diff'] = {}
        if original_corr is not None:
            # Calculate the absolute difference between original and noisy correlation
            original_noisy_diff = np.abs(original_corr - noisy_corr)

            self.metrics_dict['correlations_diff'][
                'original_noisy_correlation_diff'
            ] = original_noisy_diff

            # Calculate the absolute difference between original and denoised correlation
            original_denoised_diff = np.abs(original_corr - denoised_corr)
            self.metrics_dict['correlations_diff'][
                'original_denoised_correlation_diff'
            ] = original_denoised_diff
        else:
            self.metrics_dict['correlations_diff']['original_noisy_correlation_diff'] = None
            self.metrics_dict['correlations_diff']['original_denoised_correlation_diff'] = None

        # Calculate the absolute difference between noisy and denoised correlation
        noisy_denoised_diff = np.abs(noisy_corr - denoised_corr)
        self.metrics_dict['correlations_diff'][
            'noisy_denoised_correlation_diff'
        ] = noisy_denoised_diff

    def _calc_mean_distances_between_datsets(self) -> None:
        """
        Calculate the mean distances between original, noisy, and denoised datasets.
        This method computes the mean Euclidean distances between:
        - Original and noisy data points.
        - Original and denoised data points.
        - Noisy and denoised data points.
        The results are stored in the `metrics_dict` under the key `dataset_distances`.
        Raises:
            ValueError: If the shapes of noisy and denoised data do not match.
            ValueError: If the shapes of original and denoised data do not match
                (when original data is provided).
        Notes:
            - If `original_data` is not provided, the distances involving original data will be
                set to `None`.
            - The method assumes that the data points are stored in the 'df' key of the respective
                dictionaries.
        Updates:
            self.metrics_dict['dataset_distances']: A dictionary containing:
                - 'mean_original_noisy_distance': Mean distance between original and
                    noisy data points.
                - 'mean_original_denoised_distance': Mean distance between original and
                    denoised data points.
                - 'mean_denoised_noisy_distance': Mean distance between noisy and
                    denoised data points.
        """
        # Calculate the mean distance between original, noisy and denoised data points
        original_points = self.original_data['df'].values if 'df' in self.original_data else None
        noisy_points = self.noisy_data['df'].values
        denoised_points = self.denoised_data['df'].values

        if noisy_points.shape != denoised_points.shape:
            raise ValueError("Noisy and denoised data must have the same shape.")

        if original_points is not None and original_points.shape != denoised_points.shape:
            raise ValueError("Original and denoised data must have the same shape.")

        if 'distances' not in self.metrics_dict:
            self.metrics_dict['dataset_distances'] = {}

        if original_points is not None:
            original_noisy_distances = np.linalg.norm(original_points - noisy_points, axis=1)
            mean_original_noisy_distance = np.mean(original_noisy_distances)
            self.metrics_dict['dataset_distances'][
                'mean_original_noisy_distance'
            ] = mean_original_noisy_distance

            original_denoised_distances = np.linalg.norm(original_points - denoised_points, axis=1)
            mean_original_denoised_distance = np.mean(original_denoised_distances)
            self.metrics_dict['dataset_distances'][
                'mean_original_denoised_distance'
            ] = mean_original_denoised_distance
        else:
            self.metrics_dict['dataset_distances']['mean_original_noisy_distance'] = None
            self.metrics_dict['dataset_distances']['mean_original_denoised_distance'] = None

        # Calculate the mean distance between noisy and denoised data points
        denoised_noisy_distances = np.linalg.norm(noisy_points - denoised_points, axis=1)
        mean_denoised_noisy_distance = np.mean(denoised_noisy_distances)
        self.metrics_dict['dataset_distances'][
            'mean_denoised_noisy_distance'
        ] = mean_denoised_noisy_distance

    def calculate_metrics(self):
        """
        Calculate and evaluate various metrics for the experiment.
        This method performs the following steps:
        1. Computes auxiliary metrics such as correlation and histograms.
        2. Calculates the correlation difference between datasets.
        3. Computes the Kullback-Leibler divergence.
        4. Calculates the mean distances between datasets.
        The metrics are used to assess the relationships and differences
        between datasets in the experiment.
        """
        # Auxiliary function to calculate metrics
        self._calc_correlation()
        self._calc_histograms()

        # Calculate metrics
        self._calc_correlation_difference_between_datasets()
        self._calc_kullback_leibler_divergence()
        self._calc_mean_distances_between_datsets()

    def save_results(self):
        """
        Saves the predictions and metrics dictionaries to JSON files.
        This method converts the `predictions_dict` and `metrics_dict` attributes 
        from dictionary arrays to lists, and then writes them to JSON files in the 
        specified output path. The filenames are constructed using the `subfix_name` 
        attribute and a prefix indicating whether the model is a CNN.
        Files saved:
            - Predictions: `<subfix_name>_<cnn_prefix>predictions.json`
            - Metrics: `<subfix_name>_<cnn_prefix>metrics.json`
        Attributes:
            predictions_dict (dict): Dictionary containing prediction data.
            metrics_dict (dict): Dictionary containing metric data.
            is_cnn (bool): Indicates if the model is a CNN.
            out_path (str): Directory path where the files will be saved.
            subfix_name (str): Suffix used in the filenames.
        Raises:
            OSError: If there is an issue writing to the specified output path.
        """
        self.predictions_dict = self.dictionary_arrays_to_list(self.predictions_dict)
        self.metrics_dict = self.dictionary_arrays_to_list(self.metrics_dict)

        # Create output directory if it doesn't exist
        denoised_data_path = os.path.join(self.data_path, 'denoised')
        if not os.path.exists(denoised_data_path):
            os.makedirs(denoised_data_path)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.denoised_data['df'].to_parquet(
            os.path.join(denoised_data_path, f'{self.subfix_name}_denoised.parquet'),
            index=False
        )

        cnn_str = 'cnn_' if self.is_cnn else ''
        # Save predictions
        with open(
            os.path.join(self.out_path, f'{self.subfix_name}_{cnn_str}predictions.json'),
            'w',
            encoding='utf-8') as file:
            json.dump(self.predictions_dict, file, ensure_ascii=False, indent=4)

        # Save metrics
        with open(
            os.path.join(self.out_path, f'{self.subfix_name}_{cnn_str}metrics.json'),
            'w',
            encoding='utf-8') as file:
            json.dump(self.metrics_dict, file, ensure_ascii=False, indent=4)

    @staticmethod
    def dictionary_arrays_to_list(array_d):
        """
        Recursively converts elements in a dictionary or other data structures 
        to Python native types such as lists.
        This function processes a dictionary or other data structures containing 
        NumPy arrays, pandas Series, or other nested dictionaries, and converts 
        them into Python native types. Specifically:
          - NumPy arrays are converted to Python lists.
          - pandas Series are converted to Python lists.
          - Nested dictionaries are processed recursively.
        Args:
            array_d (dict, np.ndarray, pd.Series, or any): 
                The input data structure to be converted. It can be a dictionary 
                containing nested dictionaries, NumPy arrays, pandas Series, or 
                other data types.
        Returns:
            dict, list, or any:
                The converted data structure where NumPy arrays and pandas Series 
                are replaced with Python lists. Other data types are returned as-is.
        """
        if isinstance(array_d, dict):
            return {k: BaseExperiment.dictionary_arrays_to_list(v) for k, v in array_d.items()}
        elif isinstance(array_d, np.ndarray):
            return array_d.tolist()
        elif isinstance(array_d, pd.Series):
            return array_d.to_list()
        else:
            return array_d

    def run(self, data_file: str, y_col_name: str = '', add_noise: bool = False,
            sigma: float = 0.02, denoising_method: callable = None,
            denoising_method_params: dict = None, xai_models_params: dict = None):
        """
        Executes the main workflow of the experiment, including data loading, 
        noise addition, denoising, XAI benchmarking, metric calculation, 
        and result saving.
        Args:
            add_noise (bool, optional): Whether to add noise to the data. 
                Defaults to True.
            sigma (float, optional): The standard deviation of the noise to be added. 
                Defaults to 0.1.
            denoising_method (callable, optional): The method used for denoising the data. 
                Defaults to None.
            denoising_method_params (dict, optional): Parameters for the denoising method. 
                Defaults to None.
            xai_models_params (dict, optional): Parameters for the XAI models used in 
                benchmarking. Defaults to None.
        Raises:
            ValueError: If required data or parameters are missing during execution.
        Returns:
            None
        """
        # Load data
        self.load_data(data_file=data_file, y_col_name=y_col_name)

        # Add noise to the data if specified
        if add_noise:
            self.add_noise(sigma=sigma)

        # Denoise the data
        self.perform_denoising(denoising_method, **denoising_method_params)

        # Fit the XAI models on the original, noisy and denoised data
        if 'df' in self.original_data:
            if self.original_data['df'] is not None:
                self.perform_original_xai_benchmark(model_params=xai_models_params)
        self.perform_noisy_xai_benchmark(model_params=xai_models_params)
        self.perform_denoised_xai_benchmark(model_params=xai_models_params)

        # Calculate metrics
        self.calculate_metrics()

        # Save results
        self.save_results()


# TSExperiment #
# ------------------------------------------------------------------------------------------------ #
class TSExperiment(BaseExperiment):
    """
    TSExperiment is a subclass of BaseExperiment designed for time-series data experiments.
    Attributes:
        is_ts (bool): Indicates that this experiment is for time-series data.
    Methods:
        __init__(data_path, out_path, checkpoint_path, subfix_name, is_cnn, 
                 train_original_xai, train_noisy_xai, train_denoised_xai, train_denoising_method):
            Initializes the TSExperiment instance with paths and training configurations.
        load_data(data_file, y_col_name):
            Loads and preprocesses the time-series data from a Parquet file. The method scales 
            the data, shifts the target variable for prediction, and splits the data into 
            training and testing sets.
            Args:
                data_file (str): The name of the data file to load.
                y_col_name (str): The name of the column to be used as the target variable.
            Returns:
                tuple: A tuple containing the processed DataFrame, training features (x_train), 
                       testing features (x_test), training labels (y_train), and
                       testing labels (y_test).
    """
    def __init__(self, data_path: str, out_path: str, checkpoint_path: str, subfix_name: str,
                 is_cnn: bool = False, train_original_xai: bool = False,
                 train_noisy_xai: bool = False, train_denoised_xai: bool = False,
                 train_denoising_method: bool = False, verbose: bool = True) -> None:

        super().__init__(data_path, out_path, checkpoint_path, subfix_name, is_cnn,
                         train_original_xai, train_noisy_xai, train_denoised_xai,
                         train_denoising_method, verbose)
        self.is_ts = True

    def load_data(self, data_file: str, y_col_name: str) -> tuple:
        """
        Loads and preprocesses data from a specified file, scales the data, shifts the target
        column, and splits it into training and testing datasets.
        Args:
            data_file (str): The name of the data file to load. The file should be in
                Parquet format.
            y_col_name (str): The name of the target column in the dataset.
        Returns:
            tuple: A tuple containing the following:
                - df_data (pd.DataFrame): The preprocessed DataFrame with scaled features.
                - x_train (np.ndarray): Training input features.
                - x_test (np.ndarray): Testing input features.
                - y_train (np.ndarray): Training target values.
                - y_test (np.ndarray): Testing target values.
        Notes:
            - The target column specified by `y_col_name` is renamed to "y".
            - The target column is shifted upwards by one row to create a new column `y_shifted`.
            - The last row is dropped to handle NaN values introduced by the shift.
            - Data is scaled using `StandardScaler` from sklearn.
            - The data is split into training and testing sets with an 80-20 split,
                without shuffling.
            - The processed data is stored in the `self.original_data` dictionary for later use.
        """
        df_data = pd.read_parquet(
            os.path.join(
                self.data_path,
                data_file
            )
        )

        df_data.rename(columns={y_col_name: "y"}, inplace=True)

        # scale the data
        scaler = StandardScaler()
        df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

        # shift the target variable to create a new column
        df_data['y_shifted'] = df_data['y'].shift(-1)
        # drop the last row with NaN value
        df_data = df_data.dropna().reset_index(drop=True)
        y_shifted = df_data['y_shifted'].copy()
        df_data = df_data.drop(columns=['y_shifted'])

        # divide the data into train/test datasets
        input_vars = df_data.columns
        x_train, x_test, y_train, y_test = train_test_split(
            df_data[input_vars].values, y_shifted, test_size=0.2, shuffle=False
        )

        synthetic = 'synthetic' in data_file
        if synthetic:
            self.original_data['df'] = df_data
            self.original_data['x_train'] = x_train
            self.original_data['x_test'] = x_test
            self.original_data['y_train'] = y_train
            self.original_data['y_test'] = y_test
            self.original_data['df'].name = "original"
        else:
            self.noisy_data['df'] = df_data
            self.noisy_data['x_train'] = x_train
            self.noisy_data['x_test'] = x_test
            self.noisy_data['y_train'] = y_train
            self.noisy_data['y_test'] = y_test
            self.noisy_data['df'].name = "noisy"

        return df_data, x_train, x_test, y_train, y_test


# TabularExperiment #
# ------------------------------------------------------------------------------------------------ #
class TabularExperiment(BaseExperiment):
    """
    A class for conducting tabular data experiments, inheriting from BaseExperiment.
    Attributes:
        is_ts (bool): Indicates whether the experiment is for time series data. Defaults to False.
    Methods:
        __init__(data_path: str, out_path: str, checkpoint_path: str, subfix_name: str, 
            Initializes the TabularExperiment instance with paths and training options.
        load_data(data_file: str) -> tuple:
            Loads and preprocesses the tabular data from a specified file. Scales the data, 
            splits it into training and testing sets, and stores it in the appropriate 
            attributes based on whether the data is synthetic or noisy.
            Args:
                data_file (str): The name of the data file to load.
            Returns:
                tuple: A tuple containing the full dataframe, training inputs, testing inputs, 
                       training targets, and testing targets.
    """
    def __init__(self, data_path: str, out_path: str, checkpoint_path: str, subfix_name: str,
                 is_cnn: bool = False, train_original_xai: bool = False,
                 train_noisy_xai: bool = False, train_denoised_xai: bool = False,
                 train_denoising_method: bool = False, verbose: bool = True) -> None:
        super().__init__(data_path, out_path, checkpoint_path, subfix_name, is_cnn,
                         train_original_xai, train_noisy_xai, train_denoised_xai,
                         train_denoising_method, verbose)
        self.is_ts = False

    def load_data(self, data_file: str, y_col_name: str = '') -> tuple:
        """
        Loads data from a specified file, scales it, splits it into training and testing datasets, 
        and stores the results in the appropriate attributes based on whether the data is synthetic
            or noisy.
        Args:
            data_file (str): The name of the data file to load. The file should be in
            Parquet format.
        Returns:
            tuple: A tuple containing the following:
            - df_data (pd.DataFrame): The scaled DataFrame containing the loaded data.
            - x_train (np.ndarray): The training input features.
            - x_test (np.ndarray): The testing input features.
            - y_train (np.ndarray): The training target values.
            - y_test (np.ndarray): The testing target values.
        Side Effects:
            - Updates `self.original_data` or `self.noisy_data` attributes with the loaded and
            processed data,
              depending on whether the data is synthetic or not.
        Notes:
            - Assumes the target column is named 'y'.
            - Scales all columns in the DataFrame using `StandardScaler`.
            - Splits the data into training and testing sets with a test size of 20% and a fixed
            random state (42).
            - Determines if the data is synthetic by checking if the string 'synthetic' is in the
            `data_file` name.
        """
        df_data = pd.read_parquet(
            os.path.join(
                self.data_path,
                data_file
            )
        )

        # scale the data
        scaler = StandardScaler()
        df_data = pd.DataFrame(scaler.fit_transform(df_data.values), columns=df_data.columns)

        # divide the data into train/test datasets
        input_vars = list(set(df_data.columns) - set(['y']))
        x_train, x_test, y_train, y_test = train_test_split(
            df_data[input_vars].values, df_data['y'].values, test_size=0.2, random_state=42
        )

        synthetic = 'synthetic' in data_file
        if synthetic:
            self.original_data['df'] = df_data
            self.original_data['x_train'] = x_train
            self.original_data['x_test'] = x_test
            self.original_data['y_train'] = y_train
            self.original_data['y_test'] = y_test
            self.original_data['df'].name = "original"
        else:
            self.noisy_data['df'] = df_data
            self.noisy_data['x_train'] = x_train
            self.noisy_data['x_test'] = x_test
            self.noisy_data['y_train'] = y_train
            self.noisy_data['y_test'] = y_test
            self.noisy_data['df'].name = "noisy"

        return df_data, x_train, x_test, y_train, y_test
