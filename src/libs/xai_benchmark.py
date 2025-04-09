"""
XAI_benchmark

A class to perform model training and evaluation in the style of PyTorch Lightning. 
This class supports multiple regression models and time series models, providing 
methods for fitting, predicting, saving, and loading models.

Attributes:
    is_ts (bool): Indicates if the data is a time series.
    verbose (bool): Controls verbosity of the output.
    ridge (Ridge): Ridge regression model.
    pls (PLSRegression): Partial least squares regression model.
    decision_tree (DecisionTreeRegressor): Decision tree regression model.
    svr (SVR): Support vector regression model.
    knn (KNeighborsRegressor): K-nearest neighbors regression model.
    auto_arima (auto_arima): Auto ARIMA model for time series.
    arima (ARIMA): ARIMA model for time series.

Methods:
    __init__(is_ts: bool, model_params: dict, verbose: bool) -> None:
        Initializes the XAI_benchmark class with specified models and parameters.

    fit(X: np.array, y: np.array) -> None:
        Fits all the initialized models using the provided training data.

    predict(X: np.array, y_true: np.array = None, n_periods: int = None, 
            rolling_forecast: bool = False, get_metrics: bool = False) -> dict:
        Predicts using all the initialized models and optionally calculates metrics.

    save(path: str, subfix: str = '') -> None:
        Saves the trained models to the specified directory in pickle format.

    load(folder_path: str, must_have: str = '', subfix: str = '') -> None:
        Loads the saved models from the specified directory.
"""
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# XAI models
from sklearn.linear_model import Ridge                  # Ridge regression
from sklearn.cross_decomposition import PLSRegression   # Partial least squares regression
from sklearn.tree import DecisionTreeRegressor          # Decision tree regression
from sklearn.svm import SVR                             # Linear support vector regression
from sklearn.neighbors import KNeighborsRegressor       # K-neighbors regression
from statsmodels.tsa.arima.model import ARIMA           # ARIMA

# Local imports
from src.libs.utils import make_dir


class XAIBenchmark:
    """
    XAI_benchmark is a class designed to benchmark various machine learning models, 
    including regression models and time series models. It provides functionality 
    to initialize, fit, predict, save, and load models.
    Attributes:
        is_ts (bool): Indicates if the data is time series.
        verbose (bool): Controls verbosity of the output.
        ridge (Ridge): Ridge regression model.
        pls (PLSRegression): Partial Least Squares regression model.
        decision_tree (DecisionTreeRegressor): Decision Tree regression model.
        svr (SVR): Support Vector Regressor model.
        knn (KNeighborsRegressor): K-Nearest Neighbors regression model.
        auto_arima (auto_arima): Auto-ARIMA model for time series data.
        arima (ARIMA): ARIMA model for time series data.
    Methods:
        __init__(is_ts: bool, model_params: dict, verbose: bool):
            Initializes the XAI_benchmark class with specified models and parameters.
        fit(X: np.array, y: np.array) -> None:
            Fits all initialized models using the provided training data.
        predict(X: np.array, y_true: np.array = None, n_periods: int = None, 
                rolling_forecast: bool = False, get_metrics: bool = False) -> dict:
            Predicts using all initialized models and optionally calculates metrics.
        save(path: str, subfix: str = '') -> None:
            Saves all initialized models to the specified path in pickle format.
        load(folder_path: str, must_have: str = '', subfix: str = '') -> None:
            Loads models from the specified folder path and updates the class attributes.
    """

    def __init__(self, is_ts:bool = False, model_params:dict = None, verbose:bool = True) -> None:
        self.is_ts = is_ts
        self.verbose = verbose

        # Ridge regression
        if 'ridge' in model_params:
            if model_params['ridge']:
                self.ridge = Ridge(**model_params['ridge'])
            else:
                self.ridge = None
        else:
            self.ridge = None

        # Partial Least Squares regression
        if 'pls' in model_params:
            if model_params['pls']:
                self.pls = PLSRegression(**model_params['pls'])
            else:
                self.pls = None
        else:
            self.pls = None

        # Decision Tree regression
        if 'tree' in model_params:
            if model_params['tree']:
                self.decision_tree = DecisionTreeRegressor(**model_params['tree'])
            else:
                self.decision_tree = None
        else:
            self.decision_tree = None

        # Regression vector machine
        if 'svm' in model_params:
            if model_params['svm']:
                self.svr = SVR(**model_params['svm'])
            else:
                self.svr = None
        else:
            self.svr = None

        # K-Nearest Neighbors regression
        if 'knn' in model_params:
            if model_params['knn']:
                self.knn = KNeighborsRegressor(**model_params['knn'])
            else:
                self.knn = None
        else:
            self.knn = None

        # ARIMA model
        if self.is_ts:
            if 'arima' in model_params:
                self.arima_fit = None
                if model_params['arima']:
                    self.arima_params = model_params['arima']
                else:
                    self.arima_params = None
            else:
                self.arima = None


    def fit(self, x:np.array, y:np.array) -> None:
        """
        Fits the specified machine learning models to the provided data.
        Parameters:
        -----------
        X : np.array
            The input features for training the models.
        y : np.array
            The target values for training the models.
        Returns:
        --------
        None
        Notes:
        ------
        - This method supports multiple models including Ridge, Partial Least Squares (PLS),
          Decision Tree, Support Vector Regressor (SVR), K-Nearest Neighbors (KNN), and ARIMA.
        - If `self.verbose` is True, progress messages will be printed during the fitting process.
        - If `self.is_ts` is True, the target `y` is treated as a time series, and ARIMA fitting
          will be performed if `self.arima` is specified.
        """
        if self.ridge:
            if self.verbose:
                print('Fitting Ridge model...')
            self.ridge.fit(x, y)
        if self.pls:
            if self.verbose:
                print('Fitting Partial Least Squares model...')
            self.pls.fit(x, y)
        if self.decision_tree:
            if self.verbose:
                print('Fitting Decision Tree model...')
            self.decision_tree.fit(x, y)
        if self.svr:
            if self.verbose:
                print('Fitting Support Vector Regressor model...')
            self.svr.fit(x, y)
        if self.knn:
            if self.verbose:
                print('Fitting K-Nearest Neighbours model...')
            self.knn.fit(x, y)
        if self.is_ts:
            try:
                y = y.values
            except AttributeError:
                pass

            if self.arima_params:
                if self.verbose:
                    print('Fitting ARIMA model...')

                self.arima = ARIMA(y, **self.arima_params)
                self.arima_fit = self.arima.fit()

                # if self.verbose:
                #     print(self.arima_fit.summary())
            else:
                self.arima = None

        if self.verbose:
            print('All models fitted!')


    def predict(self, x:np.array, y_true:np.array = None, n_periods:int = None,
                rolling_forecast:bool = False, get_metrics:bool = False) -> dict:
        """
        Predicts outcomes using various models and optionally calculates evaluation metrics.
        Parameters:
        -----------
        X : np.array
            Input features for prediction.
        y_true : np.array, optional
            True target values for calculating metrics. Required if `get_metrics` is True.
        n_periods : int, optional
            Number of periods to predict for time series models (ARIMA and Auto-ARIMA).
        rolling_forecast : bool, default=False
            If True, performs rolling (step-by-step) forecasting for time series models.
        get_metrics : bool, default=False
            If True, calculates evaluation metrics for the predictions.
        Returns:
        --------
        dict
            A dictionary containing predictions for each model:
            - 'ridge': Predictions from Ridge regression model.
            - 'pls': Predictions from Partial Least Squares (PLS) model.
            - 'decision_tree': Predictions from Decision Tree model.
            - 'svm': Predictions from Support Vector Machine (SVM) model.
            - 'knn': Predictions from K-Nearest Neighbors (KNN) model.
            - 'arima': Predictions from ARIMA model (if applicable).
            - 'auto_arima': Predictions from Auto-ARIMA model (if applicable).
        dict
            A dictionary containing evaluation metrics for each model (if `get_metrics` is True):
            - 'mse': Mean Squared Error.
            - 'rmse': Root Mean Squared Error.
            - 'mae': Mean Absolute Error.
            - 'mape': Mean Absolute Percentage Error.
            - 'R2': R-squared score.
            If a model does not produce predictions, metrics will be set to None.
        Raises:
        -------
        AssertionError
            If `get_metrics` is True but `y_true` is not provided.
        """
        predictions = {
            'ridge': self.ridge.predict(x) if self.ridge else [],
            'pls': self.pls.predict(x) if self.pls else [],
            'decision_tree': self.decision_tree.predict(x) if self.decision_tree else [],
            'svm': self.svr.predict(x) if self.svr else [],
            'knn': self.knn.predict(x) if self.knn else [],
            'arima': []
        }

        # If the data is a time series and n_periods has been specified for arima models
        if n_periods and self.is_ts:
            # If the prediction will be step by step
            if self.arima_fit:
                if rolling_forecast:
                    i=0
                    while i < n_periods:
                        if self.arima_fit:
                            new_pred = self.arima_fit.forecast(steps=1)
                            predictions['arima'].append(new_pred[0])
                            self.arima.append(new_pred)
                        i+=1
                # Or all at once
                else:
                    predictions['arima'] = self.arima_fit.forecast(steps=n_periods)


        metrics = {}

        if get_metrics:
            assert y_true is not None, 'y must be provided to calculate metrics.'
            for model in ['ridge', 'pls', 'decision_tree', 'svm', 'knn', 'arima']:
                print(f'Calculating metrics for {model} model...')
                if list(predictions[model]):
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
        make_dir(path)
        names = ['ridge', 'pls', 'decision_tree', 'svr', 'knn']
        models = [self.ridge, self.pls, self.decision_tree, self.svr, self.knn]
        for name, model in zip(names, models):
            file_name = f'{name}_{subfix}.pkl' if subfix else f'{name}.pkl'
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(model, f)

        file_name = f'arima_{subfix}.pkl' if subfix else 'arima.pkl'
        if self.is_ts:
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(self.arima_fit, f)


    def load(self, folder_path:str, must_have:str='', subfix:str=''):
        """
        Save the XAI models in pickle format.

        Args:
            folder_path (str): path to saved models.
        """
        names = ['ridge', 'pls', 'decision', 'svr', 'knn', 'arima']
        loaded_models = {}

        make_dir(folder_path)

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if (os.path.isfile(full_path) and must_have in full_path and
                full_path.endswith(f'{subfix}.pkl')):
                prefix = file_name.split('_')[0]
                try:
                    i_list = names.index(prefix)
                    with open(full_path, 'rb') as f:
                        loaded_models[names[i_list]] = pickle.load(f)
                        print(f'Loaded {names[i_list]} model')
                except (pickle.UnpicklingError, FileNotFoundError, KeyError) as e:
                    print(f'Error loading model {file_name}: {e}')

        # Update attributes in self
        self.ridge = loaded_models.get('ridge', self.ridge)
        self.pls = loaded_models.get('pls', self.pls)
        self.decision_tree = loaded_models.get('decision', self.decision_tree)
        self.svr = loaded_models.get('svr', self.svr)
        self.knn = loaded_models.get('knn', self.knn)
        self.arima = loaded_models.get('arima', self.arima)
