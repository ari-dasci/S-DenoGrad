"""
This module reduces the noise level of the input data of a Neural Network
"""
from typing import Tuple, Union
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class DenoGrad():
    """
    This class encapsulates the methods needed to perform the noise reduction
    algorithm on the data associated with your neural network model.
    """

    class _DenoGradDataset(Dataset):
        """Base dataset for DenoGrad that provides index tracking."""
        def __init__(self, X: np.ndarray, Y: np.ndarray):
            self.X = X
            self.Y = Y

        def __len__(self):
            raise NotImplementedError

        def __getitem__(self, idx):
            raise NotImplementedError

    class _TabularDataset(_DenoGradDataset):
        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return idx, self.X[idx], self.Y[idx]

    class _SlidingWindowDataset(_DenoGradDataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray, window_size: int, stride: int = 1,
                     flattening: bool = False):
            super().__init__(X, Y)
            self.window_size = window_size
            self.stride = stride
            self.flattening = flattening

            # Calculate number of windows
            self.n_windows = (len(X) - window_size) // stride + 1

        def __len__(self):
            return self.n_windows

        def __getitem__(self, idx):
            # Map dataset index to original buffer index
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size

            x_window = self.X[start_idx:end_idx]

            # Target can be the value after the window or corresponding to window
            # For this unexpected case, we assume Y is aligned with X if it is a sequence
            # or Y is a target vector corresponding to the window end.
            # Simplified: Return Y corresponding to the end of the window (standard forecasting)
            # OR if Y has same length as X, return window of Y?
            # Let's assume standard "many-to-one" forecasting where Y is aligned with X's timestamps
            # If Y is (N,), then y[end_idx-1] is the target at last step.
            # For flexibility, let's return the target corresponding to the window's last step
            # unless Y is shorter (pre-processed).

            # CRITICAL FIX for compatibility:
            # If Y is supplied as full length, we take the target at the end horizon.
            # But users might provide Y already windowed? No, fit() manages raw data now.

            val_y = self.Y[start_idx:end_idx] if len(self.Y) == len(self.X) else self.Y[idx]

            if self.flattening:
                x_window = x_window.reshape(-1)

            return start_idx, x_window, val_y

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.modules.loss._Loss,
        device: torch.device = None
    ):
        """
        Initialize the DenoGrad class.

        Args:
            model (nn.Module): neural network model already trained.
            criterion (nn.modules.loss._Loss): loss function.
            device (torch.device, optional): device to run calculations on.
        """
        self._criterion: nn.modules.loss._Loss = criterion
        self._device = device
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model: nn.Module = model.to(self._device)

        # Setup model mode
        is_recurrent = any(isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)) for m in self._model.modules())
        if is_recurrent:
            self._model.train() # Keep train mode for RNN state handling usually
        else:
            self._model.eval()

        # Data Attributes (Internal)
        self._dataset: Dataset = None
        self._is_ts: bool = False
        self._is_cnn: bool = False

        first_module = next(self._model.modules())
        if isinstance(first_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._is_cnn = True

        # Exposed for plotting/debugging but managed via dataset now
        self._x_storage: np.ndarray = None 
        self._y_storage: np.ndarray = None

    def __repr__(self):
        return (f"DenoGrad(model={self._model.__class__.__name__}, "
                f"criterion={self._criterion.__class__.__name__}, "
                f"device={self._device})")



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
    def _transform(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        batch_size: int=1000,
        save_gradients: bool=True,
        denoise_y: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Generic transform loop."""
        if self._dataset is None:
            raise RuntimeError("You must call .fit() before .transform()")

        # TODO: Enforce rule?
        # No Y denoising for Time Series
        # Usually Y is future values or implicit in X
        # But it could be a sequence target
        # if self._is_ts:
        #      denoise_y = False
        initial_state_dict = copy.deepcopy(self._model.state_dict())
        x_gradient_list = []
        y_gradient_list = []
        epoch = 0
        more_gradients = 1

        # Batch size fix: cannot exceed dataset length
        batch_size = min(batch_size, len(self._dataset))

        with tqdm(total=max_epochs) as pbar:
            while epoch < max_epochs and more_gradients:
                more_gradients = 0

                # Initialize Accumulators for the Epoch (Consensus Strategy)
                # We normalize by the number of times a specific point was "visited" by a window
                grad_accum_x = np.zeros_like(self._dataset.X, dtype=np.float32)
                count_accum_x = np.zeros_like(self._dataset.X, dtype=np.float32)

                grad_accum_y = None
                count_accum_y = None
                if denoise_y:
                    grad_accum_y = np.zeros_like(self._dataset.Y, dtype=np.float32)
                    count_accum_y = np.zeros_like(self._dataset.Y, dtype=np.float32)

                loader = torch.utils.data.DataLoader(
                    self._dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0 # Avoid multiprocessing issues with in-place numpy updates if safe
                )

                for indices, x_batch, y_batch in loader:
                    x_tensor = x_batch.to(self._device).float()
                    y_tensor = y_batch.to(self._device).float()
                    x_tensor.requires_grad_(True)
                    y_tensor.requires_grad_(True)

                    self._criterion.zero_grad()

                    # 1. Forward
                    # Handle CNN Dimension Permutation (B, L, C) -> (B, C, L) if needed
                    if self._is_cnn and x_tensor.ndim == 3:
                        # Assuming input is (B, L, C) from SlidingWindow
                        out = self._model(x_tensor.transpose(1, 2))
                    else:
                        out = self._model(x_tensor)

                    # Shape Alignment
                    if out.shape != y_tensor.shape:
                        if out.ndim == y_tensor.ndim + 1:
                            out = out.squeeze(-1)
                        elif y_tensor.ndim == out.ndim + 1:
                            y_tensor = y_tensor.squeeze(-1)

                    # 2. Loss & Backward
                    loss = self._criterion(out, y_tensor)
                    loss.backward()

                    # 3. Validation & Gradient Calculation
                    y_pred_np = out.detach().cpu().numpy()
                    y_true_np = y_tensor.detach().cpu().numpy()

                    # Check shape again for broadcasting in numpy
                    if y_pred_np.shape != y_true_np.shape:
                        # Force match if squeeze needed
                        if y_pred_np.size == y_true_np.size:
                            y_true_np = y_true_np.reshape(y_pred_np.shape)

                    diff = np.abs(y_pred_np - y_true_np)
                    mask_apply = (diff > nr_threshold).astype(np.float32)
                    more_gradients += mask_apply.sum()

                    # Fix broadcasting for mask: (B, ...) -> match grad shape
                    # x_tensor.grad is (B, Window, Feat) or (B, Feat)
                    grad_x = x_tensor.grad.detach().cpu().numpy()

                    # Check if Y gradients are available for joint normalization
                    # They might not be if Y is discrete (LongTensor) or requires_grad failed
                    grad_y = None
                    try:
                        if y_tensor.grad is not None:
                            grad_y = y_tensor.grad.detach().cpu().numpy()
                    except RuntimeError:
                        pass # grad_y remains None

                    # JOINT NORMALIZATION Logic (Always preferred if grad_y exists)
                    if grad_y is not None:
                        # Flatten both to (Batch, -1) to handle any shape (Tabular or TS)
                        flat_x = grad_x.reshape(grad_x.shape[0], -1)
                        flat_y = grad_y.reshape(grad_y.shape[0], -1)

                        # Concatenate features for norm calculation
                        # This scales the step by the TOTAL steepness of the loss landscape
                        flat_all = np.concatenate([flat_x, flat_y], axis=1)
                        l2_norms = np.linalg.norm(flat_all, axis=1, keepdims=True)
                        l2_norms[l2_norms == 0] = 1e-8

                        # Apply shared norm
                        flat_x /= l2_norms
                        flat_y /= l2_norms

                        # Reshape back to original dimensions
                        grad_x = flat_x.reshape(grad_x.shape)
                        grad_y = flat_y.reshape(grad_y.shape)
                    else:
                        # Fallback: INDEPENDENT Normalization (Only X)
                        # Occurs if Y is discrete or frozen without grads
                        flat_grads_x = grad_x.reshape(grad_x.shape[0], -1)
                        l2_norms_x = np.linalg.norm(flat_grads_x, axis=1, keepdims=True)
                        l2_norms_x[l2_norms_x == 0] = 1e-8
                        flat_grads_x /= l2_norms_x
                        grad_x = flat_grads_x.reshape(grad_x.shape)

                    # Prepare adjustments

                    # 1. Adjustment for X (Always applied)
                    # Expand mask to match grad_x dimensions
                    mask_apply_x = mask_apply.copy()
                    while mask_apply_x.ndim < grad_x.ndim:
                        mask_apply_x = np.expand_dims(mask_apply_x, axis=-1)
                    adjustment_x = grad_x * nrr * mask_apply_x

                    # 2. Adjustment for Y (Only if requested and available)
                    adjustment_y = None
                    if denoise_y and grad_y is not None:
                        mask_apply_y = mask_apply.copy()
                        while mask_apply_y.ndim < grad_y.ndim:
                            mask_apply_y = np.expand_dims(mask_apply_y, axis=-1)
                        adjustment_y = grad_y * nrr * mask_apply_y

                    # 4. Accumulate Updates (Do NOT apply in-place yet)
                    indices_np = indices.numpy()

                    if self._is_ts:
                        # For Sliding Window: Accumulate into Global Buffer
                        for i, start_idx in enumerate(indices_np):
                            end_idx = start_idx + self._dataset.window_size

                            # Accumulate X
                            grad_accum_x[start_idx:end_idx] += adjustment_x[i]
                            count_accum_x[start_idx:end_idx] += 1.0

                            # Accumulate Y (if applicable)
                            if denoise_y and adjustment_y is not None:
                                if len(self._dataset.Y) == len(self._dataset.X):
                                    # Y is a sequence aligned with X
                                    grad_accum_y[start_idx:end_idx] += adjustment_y[i]
                                    count_accum_y[start_idx:end_idx] += 1.0
                                else:
                                    # Y is a single target per window
                                    target_idx = indices_np[i]
                                    grad_accum_y[target_idx] += adjustment_y[i]
                                    count_accum_y[target_idx] += 1.0
                    else:
                        # For Tabular: Direct Accumulation
                        # Note: In tabular, count is usually 1 unless batches repeat indices?
                        # Dataset doesn't repeat, but good to be generic.
                        grad_accum_x[indices_np] += adjustment_x
                        count_accum_x[indices_np] += 1.0

                        if denoise_y and adjustment_y is not None:
                            grad_accum_y[indices_np] += adjustment_y
                            count_accum_y[indices_np] += 1.0

                    if save_gradients:
                        x_gradient_list.append(grad_x)
                        if denoise_y and grad_y is not None:
                            y_gradient_list.append(grad_y)

                # END OF EPOCH UPDATE
                # Normalize accumulated gradients by the number of contributions (Consensus)

                # Avoid division by zero
                count_accum_x[count_accum_x == 0] = 1.0
                avg_adjustment_x = grad_accum_x / count_accum_x
                self._dataset.X -= avg_adjustment_x

                if denoise_y and grad_accum_y is not None:
                    count_accum_y[count_accum_y == 0] = 1.0
                    avg_adjustment_y = grad_accum_y / count_accum_y
                    self._dataset.Y -= avg_adjustment_y

                epoch += 1
                pbar.update(1)

        if epoch < max_epochs:
            print(f"converged at epoch {epoch}")

        final_state_dict = self._model.state_dict()
        weights_changed = any(not torch.equal(initial_state_dict[k],
                                              final_state_dict[k]) for k in initial_state_dict)
        if weights_changed:
            print("WARNING: Model weights CHANGED during denoising!")
        else:
            print("SUCCESS: Model weights remained UNCHANGED during denoising.")

        return self._x_storage, self._y_storage, x_gradient_list, y_gradient_list


    # Public methods
    # --------------------------------------------------------------------------
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, list, str] = None,
        is_ts: bool = False,
        window_size: int = None,
        stride: int = 1,
        flattening: bool = False
    ) -> 'DenoGrad':
        """
        Fit the model to the input data.

        Args:
            X: Input data. Can be numpy array, torch Tensor, or pandas DataFrame (if hasattr values)
            y: Target data. 
               - If X is DataFrame and y is list/str, these are column names in X to treat as target
               - Otherwise, array/tensor of targets.
            is_ts (bool): Whether data is Time Series.
            window_size (int): Size of sliding window (Required if is_ts=True).
            stride (int): Stride for sliding window.
            flattening (bool): Whether to flatten windows (e.g. for MLP on TS data).
            is_cnn (bool): Whether model requires (B, C, L) format (often for 1D CNNs).
        """
        self._is_ts = is_ts

        # 1. Uniform Data Conversion to Numpy
        def to_numpy(d):
            if d is None:
                return None
            if isinstance(d, torch.Tensor):
                return d.detach().cpu().numpy()
            if hasattr(d, 'values'):
                return d.values # Pandas support
            return np.array(d)

        # Handle Pandas "y is implicit in X" case
        if hasattr(X, 'columns') and (isinstance(y, str) or isinstance(y, list)):
            # Assume X is DataFrame
            if isinstance(y, str):
                y = [y]
            Y_np = X[y].values
            # If we want to separate features and targets from X, we can.
            # But for denoising X, we usually keep all columns in X.
            # We just need Y for the loss.
            X_np = X.values
        else:
            X_np = to_numpy(X)
            Y_np = to_numpy(y)

        if Y_np is None:
            raise ValueError("Target 'y' must be provided.")

        # Store backups (references)
        self._x_storage = X_np # We will modify this array in-place!
        self._y_storage = Y_np

        # 2. Dataset Strategy
        if is_ts:
            if window_size is None:
                raise ValueError("window_size must be provided for Time Series data.")
            self._dataset = self._SlidingWindowDataset(
                self._x_storage, self._y_storage,
                window_size=window_size,
                stride=stride,
                flattening=flattening
            )
        else:
            self._dataset = self._TabularDataset(
                self._x_storage, self._y_storage
            )

        return self


    def transform(
        self,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Decrease the noise level in the input data (x and y).
        """
        return self._transform(
             nrr, nr_threshold, max_epochs, batch_size, save_gradients, denoise_y
        )

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, list, str] = None,
        is_ts: bool = False,
        window_size: int = None,
        stride: int = 1,
        flattening: bool = False,
        nrr: float=0.05,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Fit the model to the input data and decrease the noise level in the input data (x and y).
        """
        self.fit(X, y, is_ts, window_size, stride, flattening)
        return self._transform(
             nrr, nr_threshold, max_epochs, batch_size, save_gradients, denoise_y
        )
