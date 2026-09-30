"""
This module reduces the noise level of the input data of a Neural Network
"""
from typing import Tuple, Union
from importlib.metadata import version as _pkg_version, PackageNotFoundError
import copy
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    __version__ = _pkg_version("denograd")
except PackageNotFoundError:          # ejecutado desde el repo, sin instalar
    __version__ = "0.0.0.dev0"


_TS_STRATEGIES = ("global", "window")


def _align_shapes(out: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop a trailing singleton so that model output and target line up."""
    if out.shape != y.shape:
        if out.ndim == y.ndim + 1:
            out = out.squeeze(-1)
        elif y.ndim == out.ndim + 1:
            y = y.squeeze(-1)
    return out, y


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
        # Columns of X that ARE the target (autoregressive series), or None when
        # y is a separate array. The global strategy refines them as one variable.
        self._target_idx: list = None

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
        max_epochs: int=None,
        batch_size: int=1000,
        save_gradients: bool=True,
        denoise_y: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0,
        ts_strategy: str="global",
        lam: float=1.0,
        lr: float=0.05,
        tol: float=1e-7
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Dispatch to the step rule that fits the data, and check the model."""
        if self._dataset is None:
            raise RuntimeError("You must call .fit() before .transform()")
        if ts_strategy not in _TS_STRATEGIES:
            raise ValueError(f"ts_strategy must be one of {_TS_STRATEGIES}, "
                             f"got {ts_strategy!r}")

        use_global = self._is_ts and ts_strategy == "global"

        # Each rule silently ignores the other's knobs, which is exactly how a
        # result gets produced with settings that never took effect.
        if use_global:
            ignored = {"nrr": (nrr, 0.5), "nr_threshold": (nr_threshold, 0.01),
                       "eta_x": (eta_x, 1.0), "eta_y": (eta_y, 1.0)}
        else:
            ignored = {"lam": (lam, 1.0), "lr": (lr, 0.05), "tol": (tol, 1e-7)}
        changed = [k for k, (v, d) in ignored.items() if v != d]
        if changed:
            rule = "ts_strategy='global'" if use_global else "the Gauss-Newton step"
            warnings.warn(f"{', '.join(changed)} have no effect under {rule}.",
                          stacklevel=3)

        if max_epochs is None:
            max_epochs = 300 if use_global else 100

        initial_state_dict = copy.deepcopy(self._model.state_dict())
        if use_global:
            result = self._transform_global(
                lam=lam, lr=lr, max_epochs=max_epochs, tol=tol,
                batch_size=batch_size, save_gradients=save_gradients,
                denoise_y=denoise_y
            )
        else:
            result = self._transform_gauss_newton(
                nrr=nrr, nr_threshold=nr_threshold, max_epochs=max_epochs,
                batch_size=batch_size, save_gradients=save_gradients,
                denoise_y=denoise_y, eta_x=eta_x, eta_y=eta_y
            )

        final_state_dict = self._model.state_dict()
        weights_changed = any(not torch.equal(initial_state_dict[k],
                                              final_state_dict[k]) for k in initial_state_dict)
        if weights_changed:
            print("WARNING: Model weights CHANGED during denoising!")
        else:
            print("SUCCESS: Model weights remained UNCHANGED during denoising.")

        return result


    def _transform_global(
        self,
        lam: float,
        lr: float,
        max_epochs: int,
        tol: float,
        batch_size: int,
        save_gradients: bool,
        denoise_y: bool
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Refine the whole series at once by gradient descent on

            criterion(f(windows), targets)  +  lam * mean((z - z_observed)^2)

        where z is every value being refined. The series is a single tensor and
        the windows are views of it, so one backward pass gives each timestep
        the SUM of the contributions of every window that contains it: the
        chain rule does the consensus, with no accumulator and no counter. When
        the target is a column of X (`target_cols`), it is literally the same
        variable and receives its input and target contributions together.

        With stride 1 there are T constraints for T x C unknowns, so the fit
        term alone is underdetermined; the proximity term picks, among the
        series the model finds self-consistent, the one closest to what was
        observed. Both terms are means, so `lam` keeps its meaning across series
        lengths, provided the criterion averages (reduction='mean').
        """
        ds = self._dataset
        n_win = len(ds)
        if n_win == 0:
            raise ValueError(
                f"Series too short: {len(ds.X)} steps cannot hold one window of "
                f"{ds.window_size} plus {ds.future} step(s) ahead."
            )
        shared = self._target_idx is not None
        if shared and ds.X.ndim != 2:
            raise ValueError("target_cols requires X of shape (T, n_features).")

        dev = self._device
        x_obs = torch.as_tensor(self._x_storage, dtype=torch.float32, device=dev)
        x = x_obs.clone().requires_grad_(True)
        refined = [(x, x_obs)]
        y = None
        if not shared:
            y_obs = torch.as_tensor(self._y_storage, dtype=torch.float32, device=dev)
            y = y_obs.clone().requires_grad_(denoise_y)
            if denoise_y:
                refined.append((y, y_obs))
        n_refined = sum(v.numel() for v, _ in refined)

        # Target of window s is step s*stride + window_size + future - 1, the
        # same one _SlidingWindowDataset hands the Gauss-Newton rule.
        starts = torch.arange(n_win, device=dev) * ds.stride
        tgt_idx = starts + ds.window_size + ds.future - 1
        batch_size = min(batch_size, n_win)

        opt = torch.optim.Adam([v for v, _ in refined], lr=lr)
        x_gradient_list = []
        y_gradient_list = []
        prev = None
        epoch = 0

        # The model is frozen: only the data moves. Switching its parameters off
        # spares a backward into every weight; their flags are restored after.
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        for p in trainable:
            p.requires_grad_(False)
        try:
            with tqdm(total=max_epochs) as pbar:
                while epoch < max_epochs:
                    opt.zero_grad(set_to_none=True)
                    fit_total = 0.0
                    for c0 in range(0, n_win, batch_size):
                        c1 = min(c0 + batch_size, n_win)
                        # unfold puts the window axis last: (n, C, W) -> (n, W, C)
                        wins = x.unfold(0, ds.window_size, ds.stride)[c0:c1].movedim(-1, 1)
                        if ds.flattening:
                            wins = wins.reshape(wins.shape[0], -1)
                        if self._is_cnn and wins.ndim == 3:
                            out = self._model(wins.transpose(1, 2))
                        else:
                            out = self._model(wins)
                        if shared:
                            tgt = x[tgt_idx[c0:c1]][:, self._target_idx]
                        else:
                            tgt = y[tgt_idx[c0:c1]]
                        out, tgt = _align_shapes(out, tgt)
                        # Weighted by the chunk's share, so the sum over chunks is
                        # the criterion over all windows whatever batch_size is.
                        loss_fit = self._criterion(out, tgt) * ((c1 - c0) / n_win)
                        loss_fit.backward()
                        fit_total += float(loss_fit.detach())

                    sq = sum(((v - v0) ** 2).sum() for v, v0 in refined)
                    loss_prox = lam * sq / n_refined
                    loss_prox.backward()

                    if shared and not denoise_y:
                        # The target columns are part of X, so holding them
                        # fixed means cancelling their gradient, not skipping y.
                        x.grad[:, self._target_idx] = 0.0

                    if save_gradients:
                        x_gradient_list.append(x.grad.detach().cpu().numpy().copy())
                        if y is not None and denoise_y:
                            y_gradient_list.append(y.grad.detach().cpu().numpy().copy())

                    opt.step()
                    epoch += 1
                    total = fit_total + float(loss_prox.detach())
                    pbar.set_postfix(fit=f"{fit_total:.3g}", prox=f"{float(loss_prox):.3g}")
                    pbar.update(1)
                    if prev is not None and abs(prev - total) < tol:
                        break
                    prev = total
        finally:
            for p in trainable:
                p.requires_grad_(True)

        if epoch < max_epochs:
            print(f"converged at epoch {epoch}")

        x_ref = x.detach().cpu().numpy()
        if shared:
            y_ref = x_ref[:, self._target_idx].copy()
        else:
            y_ref = y.detach().cpu().numpy()
        return x_ref, y_ref, x_gradient_list, y_gradient_list


    def _transform_gauss_newton(
        self,
        nrr: float,
        nr_threshold: float,
        max_epochs: int,
        batch_size: int,
        save_gradients: bool,
        denoise_y: bool,
        eta_x: float,
        eta_y: float
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Per-sample Gauss-Newton steps; for series, averaged over windows."""
        # Refine copies: the observed data stays as fitted, so every call to
        # transform() starts from it rather than from the previous call's output.
        self._dataset.X = self._x_storage.copy()
        self._dataset.Y = self._y_storage.copy()

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
                    out, y_tensor = _align_shapes(out, y_tensor)

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

        return self._dataset.X, self._dataset.Y, x_gradient_list, y_gradient_list


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
        flattening: bool = False,
        target_cols: Union[int, list] = None
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
            target_cols (int or list of int): positions of the columns of X
                that ARE the target, as in autoregressive forecasting where the
                target's history is an input channel. Pass it instead of `y`.
                The global time-series strategy then refines each target value
                once, as the single variable it is; a separate `y` array would
                be a second, independent copy of it. Column names given as `y`
                with a DataFrame `X` are treated the same way.
        """
        self._is_ts = is_ts
        self._target_idx = None

        # 1. Uniform Data Conversion to Numpy
        def to_numpy(d):
            if d is None:
                return None
            if isinstance(d, torch.Tensor):
                return d.detach().cpu().numpy()
            if hasattr(d, 'values'):
                return d.values # Pandas support
            return np.array(d)

        if target_cols is not None:
            if y is not None:
                raise ValueError("Pass either y or target_cols, not both.")
            X_np = to_numpy(X)
            if X_np.ndim != 2:
                raise ValueError("target_cols requires X of shape (n_samples, n_features).")
            idx = [target_cols] if np.isscalar(target_cols) else list(target_cols)
            self._target_idx = [int(i) for i in idx]
            Y_np = X_np[:, self._target_idx]
        # Handle Pandas "y is implicit in X" case
        elif hasattr(X, 'columns') and (isinstance(y, str) or isinstance(y, list)):
            # Assume X is DataFrame
            if isinstance(y, str):
                y = [y]
            Y_np = X[y].values
            # If we want to separate features and targets from X, we can.
            # But for denoising X, we usually keep all columns in X.
            # We just need Y for the loss.
            X_np = X.values
            self._target_idx = [int(i) for i in X.columns.get_indexer(y)]
        else:
            X_np = to_numpy(X)
            Y_np = to_numpy(y)

        if Y_np is None:
            raise ValueError("Target 'y' must be provided.")

        # Check matching lengths
        # In tabular, X and Y must have same number of samples
        # In TS, we assume Y is aligned with X (same temporal stamp) we'll apply future logic later
        assert len(X_np) == len(Y_np), "X and y must have the same number of samples."

        # Work on our own float copies: they are the observed data every
        # transform() starts from, and X_np/Y_np may be the caller's own arrays
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
        max_epochs: int=None,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0,
        ts_strategy: str="global",
        lam: float=1.0,
        lr: float=0.05,
        tol: float=1e-7
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Decrease the noise level in the input data (x and y).

        Every call starts from the data passed to fit(), so transform() can be
        run several times on one fit to compare settings.

        Tabular data (and series with ts_strategy='window') take per-sample
        Gauss-Newton steps, each scaled by the sample's own residual, so how far
        a point moves reflects how far it sits from what the network models.
        Series with ts_strategy='global' (the default) are refined as a whole;
        see ts_strategy below.

        Args:
            max_epochs (int): maximum number of refinement iterations. Defaults
                to 100 for the Gauss-Newton step and 300 for the global one.
            denoise_y (bool): whether to refine the target as well. Under the
                global strategy with `target_cols`, False holds the target
                columns of X fixed while the other channels move.
            batch_size (int): samples (or windows) per forward pass. Under the
                global strategy it only bounds memory: the gradient is the same
                whatever its value, up to floating-point rounding (which can
                move the `tol` stop by an iteration).
            save_gradients (bool): keep the per-iteration gradients and return
                them.
            ts_strategy (str): how a time series is refined.
                'global' (default): the series is one tensor and the windows
                are views of it, so the chain rule sums, for every timestep,
                the contributions of all the windows that contain it. It
                minimises criterion(f(windows), targets) + lam * mean((z -
                z_observed)^2) over every refined value z with Adam.
                'window': the pre-2.0 rule. Each window takes its own
                Gauss-Newton step and each timestep receives the average of the
                steps of the windows covering it. Kept to reproduce earlier
                results; it barely moves the data once the backbone fits well,
                because every step is proportional to a small residual.
                Ignored for tabular data.
            lam (float): global strategy only. Weight of the proximity term,
                which makes the underdetermined series problem well posed by
                preferring the self-consistent series closest to the observed
                one. Larger values keep the data closer to what was measured.
            lr (float): global strategy only. Adam learning rate on the data.
            tol (float): global strategy only. Stops once the total objective
                changes by less than this between iterations.
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
            nrr, nr_threshold, eta_x and eta_y belong to the Gauss-Newton step
            and have no effect under ts_strategy='global'; lam, lr and tol have
            none under the Gauss-Newton step. Setting one that does not apply
            raises a warning.
        """
        return self._transform(
            nrr=nrr, nr_threshold=nr_threshold, max_epochs=max_epochs,
            batch_size=batch_size, save_gradients=save_gradients,
            denoise_y=denoise_y, eta_x=eta_x, eta_y=eta_y,
            ts_strategy=ts_strategy, lam=lam, lr=lr, tol=tol
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
        target_cols: Union[int, list] = None,
        nrr: float=0.5,
        nr_threshold: float=0.01,
        max_epochs: int=None,
        denoise_y: bool=True,
        batch_size: int=1000,
        save_gradients: bool=True,
        eta_x: float=1.0,
        eta_y: float=1.0,
        ts_strategy: str="global",
        lam: float=1.0,
        lr: float=0.05,
        tol: float=1e-7
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Fit the model to the input data and decrease the noise level in the input data (x and y).

        See fit() for target_cols and transform() for the rest.
        """
        self.fit(X, y, is_ts=is_ts, window_size=window_size, future=future,
                 stride=stride, flattening=flattening, target_cols=target_cols)
        return self._transform(
            nrr=nrr, nr_threshold=nr_threshold, max_epochs=max_epochs,
            batch_size=batch_size, save_gradients=save_gradients,
            denoise_y=denoise_y, eta_x=eta_x, eta_y=eta_y,
            ts_strategy=ts_strategy, lam=lam, lr=lr, tol=tol
        )
