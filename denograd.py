"""
This module reduces the noise level of the input data of a Neural Network
"""
from typing import Tuple, Union
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn
from torch.utils.data import Dataset
from IPython.display import display, clear_output
from tqdm import tqdm


class DenoGrad():
    """
    This class encapsulates the methods needed to perform the noise reduction
    algorithm on the data associated with your neural network model.
    """
    class WrappedDataset(Dataset):
         def __init__(self, X, Y):
              self.X = X
              self.Y = Y
              self.shape = X.shape # Keep shape info accessible
         
         def __len__(self):
              return len(self.X)
              
         def __getitem__(self, idx):
              return self.X[idx], self.Y[idx]

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.modules.loss._Loss,
        device: torch.device = None,
        is_ts: bool = False,
        is_cnn: bool = False
    ):
        """
        Initialize the DenoGrad class.

        Args:
            model (nn.Module): neural network model already trained.
            criterion (nn.modules.loss._Loss): loss function.
            device (torch.device, optional): device to run calculations on.
                If None, it will try to use CUDA if available, else CPU.
            is_ts (bool): if the model is prepared to work with time series. Default False.
            is_cnn (bool): if the model is a CNN. Default False.
        """
        self._criterion: nn.modules.loss._Loss = criterion
        if device is None and torch.cuda.is_available():
            self._device: torch.device = torch.device('cuda')
        else:
            self._device: torch.device = torch.device('cpu')
        self._model: nn.Module = model.to(self._device)
        self._x_noisy: np.ndarray = None
        self._y_noisy: np.ndarray = None
        self.is_ts: bool = is_ts
        self.is_cnn: bool = is_cnn

    def __repr__(self):
        return (f"DenoGrad(model={self._model.__class__.__name__}, "
                f"device={self._device}, is_ts={self.is_ts}, is_cnn={self.is_cnn})")



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
            axes (matplotlib.axes.Axes): matplotlib axes object used to plot the data.
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
            axes (matplotlib.axes.Axes): matplotlib axes object used to plot the data.
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
        path_to_save_imgs: str='',
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
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

        fig = plt.figure()
        if plot_progress:
            if self._x_noisy.shape[1] == 2:
                fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 8))
            elif self._x_noisy.shape[1] == 1:
                fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            else:
                raise ValueError('The input data must have 1 or 2 features in order to plotted')

        epoch = 0
        apply_gradient = [True, True]
        with tqdm(total=max_epochs) as pbar1:
            while epoch < max_epochs and np.array(apply_gradient).any() > 0:
                x_tensor = torch.tensor(x_tensor, requires_grad=True).to(self._device)
                y_tensor = torch.tensor(y_tensor, requires_grad=True).to(self._device)

                # Calculate the gradients for X and Y performing a backpropagation step.
                self._criterion.zero_grad()

                y_predicted = self._model.forward(
                    x_tensor.float().to(self._device).view(-1, x_tensor.shape[-1])
                )
                y_predicted.requires_grad_(True)
                y_predicted.retain_grad()

                loss = self._criterion(
                    y_predicted,
                    y_tensor.float().to(self._device).view(-1, y_tensor.shape[-1])
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

                total_grad = np.concatenate((grad_l_x, grad_l_y), axis=1)
                l2_grad = np.linalg.norm(total_grad)
                grad_l_x = grad_l_x / l2_grad
                grad_l_y = grad_l_y / l2_grad

                if save_gradients:
                    x_gradient_list.append(grad_l_x)
                    y_gradient_list.append(grad_l_y)

                # Update the input data
                x_tensor = x_tensor.detach().cpu().numpy()
                y_tensor = y_tensor.detach().cpu().numpy()

                x_tensor -= grad_l_x*nrr*apply_gradient
                if denoise_y:
                    y_tensor -= grad_l_y*nrr*apply_gradient

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
                        raise ValueError(
                            'The input data must have 1 or 2 features in order to plotted'
                        )

                    if path_to_save_imgs:
                        img_name = f"{path_to_save_imgs}/grafico_{epoch}.png"
                        plt.savefig(img_name, dpi=300, bbox_inches='tight')

                    # Show the plots
                    display(fig)
                    # Clear the output
                    clear_output(wait=True)

                epoch += 1
                pbar1.update(1)

        if epoch >= max_epochs:
            print(f'Max epochs reached: {epoch}/{max_epochs}')
        else:
            print('Noise threshold reached in all data points at epoch: ', epoch)

        return x_tensor, y_tensor, x_gradient_list, y_gradient_list


    def _transform_time_series(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        batch_size: int=1000,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
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
        # Check if the model has any recurrent layers
        is_recurrent = any(isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)) for m in self._model.modules())
        if is_recurrent:
            print("Warning: The model contains recurrent layers. \
Denoising with RNNs may be slow and less effective.")
            self._model.train()
        else:
            self._model.eval()

        initial_state_dict = copy.deepcopy(self._model.state_dict())

        x_gradient_list = []
        y_gradient_list = []
        epoch = 0
        more_gradients_to_apply = 1
        # For noise reduction on inputs, batch size mainly affects VRAM usage and speed,
        # not the optimization trajectory itself (unlike training weights).
        # We cap the batch size at the number of available windows.

        batch_size = min(batch_size, self._x_noisy.shape[0])

        with tqdm(total=max_epochs) as pbar1:
            while epoch < max_epochs and more_gradients_to_apply:
                more_gradients_to_apply = 0

                loader = torch.utils.data.DataLoader(
                    self._x_noisy,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )

                # Iterate over the windows
                for i_loader , win_target in enumerate(loader):
                    windows, targets = win_target
                    x_tensor = windows.to(self._device).float()
                    y_tensor = targets.to(self._device).float()
                    x_tensor.requires_grad_(True)
                    y_tensor.requires_grad_(True)

                    # Calculate the gradients for X and Y performing a backpropagation step.
                    # Set the gradients to zero
                    self._criterion.zero_grad()

                    # Predict the target for this iteration window
                    y_predicted = self._model.forward(
                        x_tensor
                    )

                    # Ensure shapes match for loss calculation
                    if y_predicted.shape != y_tensor.shape:
                        if y_predicted.ndim == y_tensor.ndim + 1:
                            y_predicted = y_predicted.squeeze(-1)
                        if y_tensor.ndim == y_predicted.ndim + 1:
                            y_tensor = y_tensor.squeeze(-1)

                    y_predicted.requires_grad_(True)
                    y_predicted.retain_grad()

                    loss = self._criterion(
                        y_predicted,
                        y_tensor
                    )

                    loss.backward()

                    # Decide if the gradient is going to be applied or not based on the threshold
                    y_predicted_array = y_predicted.detach().cpu().numpy()
                    y_tensor_array = y_tensor.detach().cpu().numpy()

                    # Ensure arrays have the same shape before subtraction to avoid broadcasting
                    # errors
                    if y_predicted_array.shape != y_tensor_array.shape:
                        if y_predicted_array.ndim == y_tensor_array.ndim + 1:
                            y_predicted_array = y_predicted_array.squeeze(-1)
                        if y_tensor_array.ndim == y_predicted_array.ndim + 1:
                            y_tensor_array = y_tensor_array.squeeze(-1)

                    apply_gradient = np.abs(y_predicted_array - y_tensor_array)
                    apply_gradient = apply_gradient > nr_threshold
                    more_gradients_to_apply += apply_gradient.sum()

                    # Ensure apply_gradient shape is (B, 1, 1) for broadcasting against (B, W, F)
                    while apply_gradient.ndim < 3:
                        apply_gradient = np.expand_dims(apply_gradient, axis=-1)

                    # Get the calculated gradients
                    grad_l_x = x_tensor.grad.detach().cpu().numpy()

                    # Normalize gradient per sample (window) instead of per batch
                    # This ensures consistent updates regardless of batch size or other samples
                    flat_grads = grad_l_x.reshape(grad_l_x.shape[0], -1)
                    l2_grad = np.linalg.norm(flat_grads, axis=1)
                    
                    # Avoid division by zero
                    l2_grad[l2_grad == 0] = 1e-8
                    
                    # Reshape for broadcasting: (B,) -> (B, 1, 1)
                    l2_grad = l2_grad.reshape(-1, 1, 1)
                    
                    grad_l_x /= l2_grad

                    if self.is_cnn:
                        # (B, F, W) -> (B, W, F)
                        grad_l_x = grad_l_x.transpose(0, 2, 1)

                    # Expand dimensions of apply_gradient to match grad_l_x for broadcasting
                    # apply_gradient usually (B, 1, 1) or (B, O).
                    # grad_l_x is (B, W, F).
                    # Broadcasting should handle (B, 1, 1) against (B, W, F).
                    # Ensure nrr is applied
                    grad_l_x = grad_l_x * nrr * apply_gradient

                    # Correct update for 3D windowed tensor (Independent Windows)
                    # grad_l_x: (Batch, Window, Feat)
                    # self._x_noisy.X: (TotalWindows, Window, Feat)
                    
                    # Re-calculate start_indices for batch
                    current_batch_size = grad_l_x.shape[0]
                    start_indices = np.arange(current_batch_size) + i_loader * batch_size

                    # Ensure apply_gradient matches broadcastable shape
                    # grad_l_x is (B, W, F). apply_gradient (after unsqueeze loop) is (B, 1, 1).
                    
                    # Calculate adjustment
                    adjustment = grad_l_x * nrr * apply_gradient
                    
                    # Vectorized in-place update
                    # start_indices corresponds to the index of the window in the full dataset
                    self._x_noisy.X[start_indices] -= adjustment

                    if save_gradients:
                        x_gradient_list.append(grad_l_x)

                epoch += 1
                pbar1.update(1)
        if epoch >= max_epochs:
            print(f'Max epochs reached: {epoch}/{max_epochs}')
        else:
            print('Noise threshold reached in all data points.')

        weights_changed = False
        current_state_dict = self._model.state_dict()
        for key in initial_state_dict:
            if not torch.equal(initial_state_dict[key], current_state_dict[key]):
                weights_changed = True
                break
        
        if weights_changed:
            print("WARNING: Model weights CHANGED during denoising!")
        else:
            print("SUCCESS: Model weights remained UNCHANGED during denoising.")

        return self._x_noisy.X, self._x_noisy.Y, x_gradient_list, y_gradient_list


    # Public methods
    # --------------------------------------------------------------------------
    def fit(self, x: Union[np.ndarray, Dataset, torch.Tensor], y: Union[np.ndarray, torch.Tensor] = None) -> 'DenoGrad':
        """
        Fit the model to the input data.

        Args:
            x (Union[np.ndarray, Dataset, pd.DataFrame, pd.Series, torch.Tensor]): 
                The training input samples.
            y (Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]): 
                The target values (real numbers).
        
        Returns:
            self: returns an instance of self.
        """
        # Helper function to convert versatile inputs to numpy or Dataset
        def convert_input(data):
            
            if data is None:
                return None
            
            # Helper: Check Tensor first to avoid false positives with .values on some PyTorch versions/types
            if isinstance(data, torch.Tensor):
                return np.array(data.detach().cpu())
            
            if hasattr(data, 'values'): # Pandas DataFrame or Series
                return data.values
                
            if isinstance(data, (np.ndarray, Dataset)):
                return data
            
            # Fallback debug
            print(f"DEBUG: Unsupported type in convert_input: {type(data)}")
            raise TypeError(f"Unsupported type {type(data)}")

        x_converted = convert_input(x)
        y_converted = convert_input(y)

        if y_converted is not None:
             # Standard case: X and y provided separately
            self._y_noisy = y = y_converted.copy() if isinstance(y_converted, np.ndarray) else y_converted
            self._x_noisy = x_converted.copy() if isinstance(x_converted, np.ndarray) else x_converted
            
            # If TS, wrap them in a Dataset that yields (x, y) so DataLoader works as expected in _transform_time_series
            if self.is_ts and isinstance(self._x_noisy, np.ndarray) and isinstance(self._y_noisy, np.ndarray):
                 self._x_noisy = self.WrappedDataset(self._x_noisy, self._y_noisy)

        else:
            assert self.is_ts, "Model expected static tabular data (is_ts=False) but 'y' \
                was NOT provided."
            # For Time Series (often Dataset objects), we usually assign directly or copy if numpy
            if hasattr(x_converted, 'copy') and not isinstance(x_converted, Dataset):
                 self._x_noisy = x_converted.copy()
            else:
                 self._x_noisy = x_converted

        return self


    def transform(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        plot_progress: bool=False,
        denoise_y: bool=True,
        path_to_save_imgs: str='',
        batch_size: int=1000,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
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
                batch_size=batch_size,
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
        path_to_save_imgs: str='',
        batch_size: int=1000,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
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
            batch_size=batch_size,
            save_gradients=save_gradients
        )

        return x_denoised, y_denoised, x_gradient_list, y_gradient_list
