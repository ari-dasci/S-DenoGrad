"""
This module reduces the noise level of the input data of a Neural Network
"""
from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn
from torch.utils.data import Dataset
from IPython.display import display, clear_output
from tqdm import tqdm


class DLNoiseReduction():
    """
    This class encapsulates the methods needed to perform the noise reduction
    algorithm on the data associated with your neural network model.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.modules.loss._Loss,
        device: torch.device = None,
        is_ts: bool = False,
        is_cnn: bool = False
    ):
        """
        Initialize the DLNoiseReduction class.

        Args:
            model (nn.Module): neural network model already trained.
            criterion (nn.modules.loss._Loss): loss function.
            is_ts (bool): if the model is prepared to work with time series. Deafult False.
        """
        self._model = model
        self._criterion = criterion
        self._device = torch.device('cuda') if device is None else device
        self._x_noisy = None
        self._y_noisy = None
        self.is_ts = is_ts
        self.is_cnn = is_cnn


    # Getters
    # --------------------------------------------------------------------------
    @property
    def model(self) -> nn.Module:
        """
        Get the neural network model.

        Returns:
            nn.Module: neural network model.
        """
        return self._model


    @property
    def criterion(self) -> nn.modules.loss._Loss:
        """
        Get the loss function.

        Returns:
            nn.modules.loss._Loss: loss function.
        """
        return self._criterion


    @property
    def device(self) -> torch.device:
        """
        Get the device where the model is going to be trained.

        Returns:
            torch.device: device.
        """
        return self._device


    @property
    def x_noisy(self) -> np.ndarray:
        """
        Get the original X input data.

        Returns:
            np.ndarray: original X input data.
        """
        return self._x_noisy


    @property
    def y_original(self) -> np.ndarray:
        """
        Get the original y input data.

        Returns:
            np.ndarray: original y input data.
        """
        return self._y_noisy


    # Setters
    # --------------------------------------------------------------------------
    @model.setter
    def model(self, model: nn.Module) -> None:
        """
        Set the neural network model.

        Args:
            model (nn.Module): neural network model.
        """
        self._model = model


    @criterion.setter
    def criterion(self, criterion: nn.modules.loss._Loss) -> None:
        """
        Set the loss function.

        Args:
            criterion (nn.modules.loss._Loss): loss function.
        """
        self._criterion = criterion


    @device.setter
    def device(self, device: torch.device) -> None:
        """
        Set the device where the model is going to be trained.

        Args:
            device (torch.device): device.
        """
        self._device = device


    @x_noisy.setter
    def x_noisy(self, x_noisy: np.ndarray) -> None:
        """
        Set the original X input data.

        Args:
            x_noisy (np.ndarray): original X input data.
        """
        self._x_noisy = x_noisy


    @y_original.setter
    def y_original(self, y_original: np.ndarray) -> None:
        """
        Set the original y input data.

        Args:
            y_original (np.ndarray): original y input data.
        """
        self._y_noisy = y_original


    # Private methods
    # --------------------------------------------------------------------------
    def _plot2D(self, axes: Axes, x: np.ndarray, y: np.ndarray) -> None:
        """
        Plot the denoised data along with the original data in a 2D plot.

        Args:
            axes (Axes): matplotlib axes object used to plot the data.
            x (np.ndarray): denoised X input data.
            y (np.ndarray): donoised y input data.
        """
        axes[0].scatter(
            self._x_noisy,
            self._y_noisy,
            color='b',
            label='Original noisy data',
            s=5,
            alpha=0.05
        )
        axes[0].legend()
        axes[0].set_title('Original Data')
        axes[1].scatter(
            x,
            y,
            color='g',
            label='Noise-reduced data',
            s=5,
            alpha=0.05
        )
        axes[1].legend()
        axes[1].set_title('Noise Reduction Progress')


    def _plot3D(self, axes: Axes, x: np.ndarray, y: np.ndarray) -> None:
        """
        Plot the denoised data along with the original data in a 3D plot.

        Args:
            axes (Axes): matplotlib axes object used to plot the data.
            x (np.ndarray): denoised X input data.
            y (np.ndarray): donoised y input data.
        """
        axes[0].scatter(
            self._x_noisy[:,0],
            self._x_noisy[:,1],
            self._y_noisy,
            marker='o',
            c=self._y_noisy,
            cmap='magma',
            alpha=0.5
        )
        axes[1].scatter(
            x[:,0],
            x[:,1],
            y,
            marker='o',
            c=y,
            cmap='viridis',
            alpha=0.5
        )


    def _transform_tabular(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        plot_progress: bool=False,
        denoise_y: bool=True,
        path_to_save_imgs: str=None,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decrease the noise level in the input data (x and y).
        If plot_progress is True, the process will take considerably more time.

        Args:
            nrr (float): noise reduction rate. Default 0.0005.
            nr_threshold (float): if the difference between the f(x') and y is
                less than nr_threshold the gradient will be applied no more. Default 0.01.
            max_epochs (int): maximum number of epochs. Default 100.
            plot_progress (bool): whether to plot the noise reduction progress or not.
                Default False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: noise-reduced input data.
        """
        x_tensor = self._x_noisy.copy()
        y_tensor = self._y_noisy.copy()
        x_gradient_list = []
        y_gradient_list = []

        if plot_progress:
            if self._x_noisy.shape[1] == 2:
                fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 8))
            elif self._x_noisy.shape[1] == 1:
                fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            else:
                raise ValueError('The input data must have 1 or 2 features in order to plotted')

        epoch = 0
        apply_gradient = [True, True]
        while epoch < max_epochs and np.array(apply_gradient).any() > 0:
            x_tensor = torch.tensor(x_tensor, requires_grad=True)
            y_tensor = torch.tensor(y_tensor, requires_grad=True)

            # Calculate the gradients for X and Y performing a backpropagation step.
            self._criterion.zero_grad()

            y_predicted = self._model.forward(
                x_tensor.float().to(self._device)
            )
            y_predicted.requires_grad_(True)
            y_predicted.retain_grad()

            loss = self._criterion(
                y_predicted,
                y_tensor.float().to(self._device)
            )
            loss.backward()

            # Decide if the gradient is going to be applied or not
            y_predicted_array = y_predicted.detach().cpu().numpy()
            y_tensor_array = y_tensor.detach().cpu().numpy()
            apply_gradient = np.abs(y_predicted_array - y_tensor_array)
            apply_gradient = apply_gradient > nr_threshold

            # Get the calculated gradients
            grad_l_x = x_tensor.grad.detach().cpu().numpy()
            grad_l_y = y_tensor.grad.detach().cpu().numpy()

            # Update the input data
            x_tensor = x_tensor.detach().cpu().numpy()
            y_tensor = y_tensor.detach().cpu().numpy()

            total_grad = np.concatenate((grad_l_x, grad_l_y), axis=1)
            l2_grad = np.linalg.norm(total_grad)
            grad_l_x = grad_l_x / l2_grad
            grad_l_y = grad_l_y / l2_grad

            if save_gradients:
                x_gradient_list.append(grad_l_x)
                y_gradient_list.append(grad_l_y)

            x_tensor -= grad_l_x*nrr*apply_gradient.sum(axis=1)[:, np.newaxis]
            if denoise_y:
                y_tensor -= grad_l_y*nrr*apply_gradient.sum(axis=1)[:, np.newaxis]

            # Plot the progression of noise reduction if specified
            if plot_progress:
                fig.suptitle(f'Epoch: {epoch} - Noise Reduction Progress')

                # Clear the plots
                axes[0].clear()
                axes[1].clear()

                # Plot the data
                if self._x_noisy.shape[1] == 2:
                    self._plot3D(axes, x_tensor, y_tensor)
                elif self._x_noisy.shape[1] == 1:
                    self._plot2D(axes, x_tensor, y_tensor)
                else:
                    raise ValueError('The input data must have 1 or 2 features in order to plotted')

                if path_to_save_imgs:
                    img_name = f"{path_to_save_imgs}/grafico_{epoch}.png"
                    plt.savefig(img_name, dpi=300, bbox_inches='tight')

                # Show the plots
                display(fig)
                # Clear the output
                clear_output(wait=True)

            epoch += 1

        if epoch >= max_epochs:
            print(f'Max epochs reached: {epoch/max_epochs}')
        else:
            print('Noise threshold reached in all data points.')

        return x_tensor, y_tensor, x_gradient_list, y_gradient_list


    def _transform_time_series(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        denoise_y: bool=True,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decrease the noise level in the input data (x and y).
        If plot_progress is True, the process will take considerably more time.

        Args:
            nrr (float): noise reduction rate. Default 0.0005.
            nr_threshold (float): if the difference between the f(x') and y is
                less than nr_threshold the gradient will be applied no more. Default 0.01.
            max_epochs (int): maximum number of epochs. Default 100.
            plot_progress (bool): whether to plot the noise reduction progress or not.
                Default False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: noise-reduced input data.
        """
        # Accelerate the runtime by finding the best cuda configuration
        torch.backends.cudnn.benchmark = True
        try:
            self._model.lstm.flatten_parameters() # compact weights to reduce memory usage.
        except RuntimeError:
            pass

        x_gradient_list = []
        y_gradient_list = []
        self._model.train() # RNN backward allowed.
        epoch = 0
        apply_gradient = [True, True]
        with tqdm(total=max_epochs*len(self._x_noisy)) as pbar1:
            while epoch < max_epochs and sum(apply_gradient) > 0:
                n_window = 0
                # Iterate over the windows
                for i, _ in enumerate(self._x_noisy):
                    # Add a dimension to match the model requirements for time series
                    # (batch, window, variables) and make it a tensor.
                    x_tensor = torch.tensor(self._x_noisy[i][0]).unsqueeze(0)
                    x_tensor.requires_grad_(True)
                    y_tensor = torch.tensor(self._x_noisy[i][1]).unsqueeze(0)
                    y_tensor.requires_grad_(True)

                    # Calculate the gradients for X and Y performing a backpropagation step.
                    # Set the gradients to zero
                    self._criterion.zero_grad()

                    # Predict the target for this iteration window
                    y_predicted = self._model.forward(
                        x_tensor.float().to(self._device)
                    )

                    # Add a dimension to match the shape of the y_tensor
                    y_predicted = y_predicted.unsqueeze(0)
                    loss = self._criterion(
                        y_predicted,
                        y_tensor.float().to(self._device)
                    )

                    loss.backward()

                    # Decide if the gradient is going to be applied or not based on the threshold
                    y_predicted_array = y_predicted.detach().cpu().numpy()
                    y_tensor_array = y_tensor.detach().cpu().numpy()
                    apply_gradient = np.abs(y_predicted_array - y_tensor_array)
                    apply_gradient = apply_gradient > nr_threshold

                    # Get the calculated gradients
                    grad_l_x = x_tensor.grad.detach().cpu().numpy()
                    grad_l_y = y_tensor.grad.detach().cpu().numpy()

                    # Get the shape of the window. In the last iteration it can be smaller.
                    window_size = grad_l_x.shape[1]
                    grad_l_y = np.tile(grad_l_y, (1, window_size, 1))
                    total_grad = np.concatenate((grad_l_x, grad_l_y), axis=2)
                    l2_grad = np.linalg.norm(total_grad)

                    if l2_grad:
                        grad_l_x /= l2_grad
                        grad_l_y /= l2_grad

                    apply_gradient = apply_gradient.squeeze(axis=0)

                    grad_l_x = grad_l_x.squeeze(axis=0)
                    grad_l_x = grad_l_x*nrr*apply_gradient
                    grad_l_x_shape = grad_l_x.shape[0]
                    if self.is_cnn:
                        grad_l_x_shape = grad_l_x.shape[1]
                        grad_l_x = grad_l_x.T

                    if save_gradients:
                        x_gradient_list.append(grad_l_x)
                    self._x_noisy.X[n_window:n_window+grad_l_x_shape] -= grad_l_x

                    if denoise_y:
                        grad_l_y = grad_l_y.mean()
                        grad_l_y = grad_l_y*nrr*apply_gradient

                        if save_gradients:
                            y_gradient_list.append(grad_l_y)
                        self._x_noisy.Y[n_window:n_window+grad_l_y.shape[0]] -= grad_l_y

                    n_window += 1
                    pbar1.update(1)

                epoch += 1
        if epoch >= max_epochs:
            print(f'Max epochs reached: {epoch/max_epochs}')
        else:
            print('Noise threshold reached in all data points.')

        return self._x_noisy.X, self._x_noisy.Y, x_gradient_list, y_gradient_list


    # Public methods
    # --------------------------------------------------------------------------
    def fit(self, x: Union[np.array, Dataset], y: np.array = None) -> None:
        """
        Fit the model to the input data.

        Args:
            x (np.array): array-like of shape (n_samples, n_features).
                The training input samples.
            y (np.array): array-like of shape (n_samples, n_targets).
                The target values (real numbers).
        """
        if y is not None:
            assert not self.is_ts, 'Model set to work with time series but «y» has been provided.'
            self._y_noisy = y.copy()
            self._x_noisy = x.copy()
        else:
            assert self.is_ts, 'Model prepared to work with tabular data but no «y» has been \
                provided.'
            self._x_noisy = x


    def transform(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        plot_progress: bool=False,
        denoise_y: bool=True,
        path_to_save_imgs: str=None,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decrease the noise level in the input data (x and y).
        If plot_progress is True, the process will take considerably more time.

        Args:
            nrr (float): noise reduction rate. Default 0.0005.
            nr_threshold (float): if the difference between the f(x') and y is
                less than nr_threshold the gradient will be applied no more. Default 0.01.
            max_epochs (int): maximum number of epochs. Default 100.
            plot_progress (bool): whether to plot the noise reduction progress or not.
                Default False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: noise-reduced input data.
        """
        x_tensor = None
        y_tensor = None
        x_gradient_list = None
        y_gradient_list = None
        if not self.is_ts:
            x_tensor, y_tensor, x_gradient_list, y_gradient_list = self._transform_tabular(
                nrr=nrr,
                nr_threshold=nr_threshold,
                max_epochs=max_epochs,
                plot_progress=plot_progress,
                denoise_y=denoise_y,
                path_to_save_imgs=path_to_save_imgs,
                save_gradients=save_gradients
            )
        else:
            x_tensor, y_tensor, x_gradient_list, y_gradient_list = self._transform_time_series(
                nrr=nrr,
                nr_threshold=nr_threshold,
                max_epochs=max_epochs,
                denoise_y=denoise_y,
                save_gradients=save_gradients
            )

        return x_tensor, y_tensor, x_gradient_list, y_gradient_list


    def fit_transform(
        self,
        x: np.array,
        y: np.array,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        plot_progress: bool=False,
        path_to_save_imgs: str=None,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to the input data and decrease the noise level in the input
        data (x and y). If plot_progress is True, the process will take
        considerably more time.

        Args:
            x (np.array): array-like of shape (n_samples, n_features).
                The training input samples.
            y (np.array): array-like of shape (n_samples, n_targets).
                The target values (real numbers).
            nrr (float): noise reduction rate. Defaults to 0.0005.
            nr_threshold (float): if the difference between the f(x') and y is
                less than nr_threshold the gradient will be applied no more.
                Defaults to 0.01.
            max_epochs (int): maximum number of epochs. Defaults to 100.
            plot_progress (bool): whether to plot the noise reduction progress or not.
                Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: noise-reduced input data.
        """
        self.fit(x, y)
        x_denoised, y_denoised, x_gradient_list, y_gradient_list = self.transform(
            nrr=nrr,
            nr_threshold=nr_threshold,
            max_epochs=max_epochs,
            plot_progress=plot_progress,
            path_to_save_imgs=path_to_save_imgs,
            save_gradients=save_gradients
        )

        return x_denoised, y_denoised, x_gradient_list, y_gradient_list


    def assert_improvement(self, x_denoised: np.ndarray, y_denoised: np.ndarray,
                           x_orig: np.ndarray, y_orig: np.ndarray) -> bool:
        """
        Assert that the noise reduction process has improved the dataset based on
        model performance.

        Args:
            x_denoised (np.ndarray): denoised X input data.
            y_denoised (np.ndarray): denoised y input data.
            x_orig (np.ndarray): original X input data, before adding noise.
            y_orig (np.ndarray): original Y input data, before adding noise.

        Returns:
            bool: whether the data has improve or not.
            
        """

        x_noisy_median_error = np.median(np.abs(self._x_noisy - x_orig))
        x_denoised_median_error = np.median(np.abs(x_denoised - x_orig))
        y_noisy_median_error = np.median(np.abs(self._y_noisy - y_orig))
        y_denoised_median_error = np.median(np.abs(y_denoised - y_orig))

        print(f'X noisy median error: {x_noisy_median_error}')
        print(f'X denoised median error: {x_denoised_median_error}')
        print(f'Y noisy median error: {y_noisy_median_error}')
        print(f'Y denoised median error: {y_denoised_median_error}')

        has_improved = x_denoised_median_error < x_noisy_median_error
        has_improved = has_improved and (y_denoised_median_error < y_noisy_median_error)

        return has_improved
