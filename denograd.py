"""
This module reduces the noise level of the input data of a Neural Network
"""
from typing import Tuple, Union
from importlib.metadata import version as _pkg_version, PackageNotFoundError
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    __version__ = _pkg_version("denograd")
except PackageNotFoundError:          # ejecutado desde el repo, sin instalar
    __version__ = "0.0.0.dev0"


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
        def __init__(self, X: np.ndarray, Y: np.ndarray, window_size: int, future: int = 1,
                     stride: int = 1, flattening: bool = False):
            super().__init__(X, Y)
            self.window_size = window_size
            self.future = future
            self.stride = stride
            self.flattening = flattening

            # Calculate number of windows
            # The last window must allow for 'future' steps ahead in Y
            # Last target index = start_idx + window_size + future - 1
            # Must be < len(Y)
            limit = len(Y) - window_size - future + 1
            if limit <= 0:
                self.n_windows = 0
            else:
                self.n_windows = (limit - 1) // stride + 1

        def __len__(self):
            return self.n_windows

        def __getitem__(self, idx):
            # Map dataset index to original buffer index
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size

            x_window = self.X[start_idx:end_idx]

            # Target is future steps ahead
            # If future=1, it is the immediate next value (index = end_idx)
            # General: end_idx + future - 1
            # (since end_idx is exclusive bound of window, it points to next element)
            # Correction: end_idx points to the element at t+1 if window is 0..t
            # Wait, slice 0:3 is indices 0,1,2. end_idx=3.
            # Next element is index 3.
            # So if future=1, index=3. => end_idx + 1 - 1 = end_idx. Correct.
            val_y = self.Y[end_idx + self.future - 1]

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

        # First *leaf* layer: model.modules() yields the container itself first,
        # so we skip anything that still has children.
        first_layer = next(
            (m for m in self._model.modules() if not list(m.children())), None
        )
        if isinstance(first_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
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
        return self._x_storage


    @property
    def y_original(self) -> np.ndarray:
        """
        Get the original y input data.

        Returns:
            np.ndarray: original y input data.
        """
        return self._y_storage


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
        self._x_storage = x_noisy


    @y_original.setter
    def y_original(self, y_original: np.ndarray) -> None:
        """
        Set the original y input data.

        Args:
            y_original (np.ndarray): original y input data.
        """
        self._y_storage = y_original


    # Private methods
    # --------------------------------------------------------------------------
    def _transform(
        self,
        nrr: float=0.5,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        batch_size: int=1000,
        save_gradients: bool=True,
        denoise_y: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0
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

                    self._model.zero_grad(set_to_none=True)

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
                    # Per-SAMPLE mask: the joint normalization below works on a
                    # whole sample at a time, so the decision to correct one must
                    # be too. With multi-output targets an element-wise mask has
                    # shape (B, n_targets) and cannot broadcast against grad_x.
                    if diff.ndim > 1:
                        sample_hit = (diff > nr_threshold).any(
                            axis=tuple(range(1, diff.ndim))
                        )
                    else:
                        sample_hit = diff > nr_threshold
                    mask_apply = sample_hit.astype(np.float32)
                    more_gradients += mask_apply.sum()

                    # Fix broadcasting for mask: (B, ...) -> match grad shape
                    # x_tensor.grad is (B, Window, Feat) or (B, Feat)
                    grad_x = x_tensor.grad.detach().cpu().numpy().copy()

                    # Check if Y gradients are available for joint normalization
                    # They might not be if Y is discrete (LongTensor) or requires_grad failed
                    grad_y = None
                    try:
                        if y_tensor.grad is not None:
                            grad_y = y_tensor.grad.detach().cpu().numpy().copy()
                    except RuntimeError:
                        pass # grad_y remains None

                    # Per-sample residual norm, used as the Gauss-Newton step length
                    resid_norm = np.linalg.norm(
                        diff.reshape(diff.shape[0], -1), axis=1, keepdims=True
                    )

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

                        # Norm of the Y block BEFORE the shared division: needed to
                        # recover the sensitivity scale for the Gauss-Newton step.
                        y_grad_norms = np.linalg.norm(flat_y, axis=1, keepdims=True)

                        # Apply shared norm
                        flat_x /= l2_norms
                        flat_y /= l2_norms

                        # Reshape back to original dimensions
                        grad_x = flat_x.reshape(grad_x.shape)
                        grad_y = flat_y.reshape(grad_y.shape)

                        # Gauss-Newton step length: the distance from the sample
                        # to the consistency surface {out(x) = y}, measured along
                        # the (unit) gradient direction:
                        #     dist = |r| / ||g||,   g = (dout/dx, -1)
                        # For any loss that is a function of the residual,
                        # grad = k * g and grad_y = k * (-1), so
                        #     ||g|| = ||grad|| / ||grad_y||
                        # and the loss constant k cancels out entirely: this needs
                        # no knowledge of the criterion, its reduction, or N.
                        # Since ||grad|| >= ||grad_y||, dist <= |r|, so a sample
                        # can never travel further than its own residual.
                        step_len = (nrr * resid_norm * y_grad_norms
                                    / l2_norms).ravel()
                    else:
                        # Fallback: INDEPENDENT Normalization (Only X)
                        # Occurs if Y is discrete or frozen without grads
                        flat_grads_x = grad_x.reshape(grad_x.shape[0], -1)
                        l2_norms_x = np.linalg.norm(flat_grads_x, axis=1, keepdims=True)
                        l2_norms_x[l2_norms_x == 0] = 1e-8
                        flat_grads_x /= l2_norms_x
                        grad_x = flat_grads_x.reshape(grad_x.shape)
                        # Without grad_y the sensitivity scale cannot be recovered,
                        # so this path keeps the fixed-length step.
                        step_len = np.full(grad_x.shape[0], nrr, dtype=np.float32)

                    # Prepare adjustments

                    # 1. Adjustment for X (Always applied)
                    # mask_apply_x stays a plain 0/1 indicator: it doubles as the
                    # consensus counter below, so the step length must not leak
                    # into it.
                    mask_apply_x = mask_apply.copy()
                    step_x = step_len.copy()
                    while mask_apply_x.ndim < grad_x.ndim:
                        mask_apply_x = np.expand_dims(mask_apply_x, axis=-1)
                        step_x = np.expand_dims(step_x, axis=-1)
                    adjustment_x = grad_x * step_x * mask_apply_x * eta_x

                    # 2. Adjustment for Y (Only if requested and available)
                    adjustment_y = None
                    if denoise_y and grad_y is not None:
                        mask_apply_y = mask_apply.copy()
                        step_y = step_len.copy()
                        while mask_apply_y.ndim < grad_y.ndim:
                            mask_apply_y = np.expand_dims(mask_apply_y, axis=-1)
                            step_y = np.expand_dims(step_y, axis=-1)
                        adjustment_y = grad_y * step_y * mask_apply_y * eta_y

                    # 4. Accumulate Updates (Do NOT apply in-place yet)
                    indices_np = indices.numpy()

                    if self._is_ts:
                        # For Sliding Window: Accumulate into Global Buffer
                        for i, start_idx in enumerate(indices_np):
                            end_idx = start_idx + self._dataset.window_size

                            # Accumulate X. With flattening=True the gradient
                            # comes back as (window_size * n_features,), so it
                            # has to be folded back into the buffer's layout.
                            adj_i = adjustment_x[i]
                            slot_shape = grad_accum_x[start_idx:end_idx].shape
                            if adj_i.shape != slot_shape:
                                adj_i = adj_i.reshape(slot_shape)
                            grad_accum_x[start_idx:end_idx] += adj_i
                            count_accum_x[start_idx:end_idx] += mask_apply_x[i]

                            # Accumulate Y (if applicable)
                            if denoise_y and adjustment_y is not None:
                                # _SlidingWindowDataset always yields a single Y
                                # target per window, never a sequence.
                                target_idx = (start_idx + self._dataset.window_size
                                              + self._dataset.future - 1)
                                if 0 <= target_idx < len(grad_accum_y):
                                    grad_accum_y[target_idx] += adjustment_y[i]
                                    count_accum_y[target_idx] += mask_apply_y[i]
                    else:
                        # For Tabular: Direct Accumulation
                        # Note: In tabular, count is usually 1 unless batches repeat indices?
                        # Dataset doesn't repeat, but good to be generic.
                        grad_accum_x[indices_np] += adjustment_x
                        count_accum_x[indices_np] += mask_apply_x

                        if denoise_y and adjustment_y is not None:
                            grad_accum_y[indices_np] += adjustment_y
                            count_accum_y[indices_np] += mask_apply_y

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
        future: int = 1,
        stride: int = 1,
        flattening: bool = False
    ) -> 'DenoGrad':
        """
        Fit the model to the input data.

        Args:
            X: Input data. Can be numpy array, torch Tensor, or pandas DataFrame (if hasattr values)
            y: Target data or Column specification. 
               - If X is DataFrame and y is list/str, these are column names in X to treat as target
               - Otherwise, array/tensor of targets.
            is_ts (bool): Whether data is Time Series.
            window_size (int): Size of sliding window (Required if is_ts=True).
            future (int): Steps ahead to predict (Required if is_ts=True). Default 1.
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

        # Check matching lengths
        # In tabular, X and Y must have same number of samples
        # In TS, we assume Y is aligned with X (same temporal stamp) we'll apply future logic later
        assert len(X_np) == len(Y_np), "X and y must have the same number of samples."

        # Work on our own float copies: the denoising loop mutates these
        # buffers in place, and X_np/Y_np may be the caller's own arrays
        # (or of an integer dtype that cannot absorb a float update).
        self._x_storage = np.array(X_np, dtype=np.float32)
        self._y_storage = np.array(Y_np, dtype=np.float32)

        # 2. Dataset Strategy
        if is_ts:
            if window_size is None:
                raise ValueError("window_size must be provided for Time Series data.")
            self._dataset = self._SlidingWindowDataset(
                self._x_storage, self._y_storage,
                window_size=window_size,
                future=future,
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
        nrr: float=0.5,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Decrease the noise level in the input data (x and y).

        Each step is scaled by the sample's own residual, so how far a point
        moves reflects how far it sits from what the network models.

        Args:
            nrr (float): noise reduction rate. The fraction of the distance to
                the model's consistency surface that a sample closes per epoch,
                so it is dimensionless: nrr=1.0 lands on the surface in a single
                step. Values between 0.3 and 0.9 converge in a handful of epochs.
            nr_threshold (float): samples whose residual falls below this are
                left alone; the loop stops once no sample exceeds it.
            eta_x, eta_y (float): per-block multipliers on the correction. Only
                their RATIO changes where the refinement lands: scaling both by
                the same factor reaches the same fixed point in more (or fewer)
                epochs, exactly as nrr does. The default 1.0/1.0 splits the step
                as ||df/dx||^2 : 1 between features and target, which is the
                maximum-likelihood attribution under isotropic Gaussian noise.
                Raise eta_x/eta_y to push more of the correction onto X, at the
                cost of overshooting the one direction the backbone can observe.
        """
        return self._transform(
            nrr=nrr, nr_threshold=nr_threshold, max_epochs=max_epochs,
            batch_size=batch_size, save_gradients=save_gradients,
            denoise_y=denoise_y, eta_x=eta_x, eta_y=eta_y
        )

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, list, str] = None,
        is_ts: bool = False,
        window_size: int = None,
        future: int = 1,
        stride: int = 1,
        flattening: bool = False,
        nrr: float=0.5,
        nr_threshold: float=0.01,
        max_epochs: int=100,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Fit the model to the input data and decrease the noise level in the input data (x and y).

        See transform() for the meaning of nrr, nr_threshold, eta_x and eta_y.
        """
        self.fit(X, y, is_ts=is_ts, window_size=window_size, future=future,
                 stride=stride, flattening=flattening)
        return self._transform(
            nrr=nrr, nr_threshold=nr_threshold, max_epochs=max_epochs,
            batch_size=batch_size, save_gradients=save_gradients,
            denoise_y=denoise_y, eta_x=eta_x, eta_y=eta_y
        )
