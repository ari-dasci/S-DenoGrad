# DenoGrad: A Model-Agnostic Framework for Gradient-Based Data Refinement

[![PyPI version](https://badge.fury.io/py/denograd.svg)](https://badge.fury.io/py/denograd)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**DenoGrad** is a novel, model-agnostic framework designed to reduce noise in both input features and target variables by leveraging the gradients of a pre-trained Deep Learning model.

In the Data-Centric AI paradigm, traditional denoising often compromises data integrity by aggressively smoothing features. DenoGrad resolves this by leveraging the **semantic spectral bias** of neural networks. Instead of requiring clean ground truth data, it freezes the weights of your predictive backbone and iteratively backpropagates error corrections into the input space, effectively shifting noisy instances toward the learned data manifold.

### Key Capabilities
* **Model-Agnostic:** Works with any differentiable PyTorch model (MLP, LSTM, CNN, Transformers, TabPFN, etc.).
* **No Clean Ground Truth Required:** Operates via self-supervised input optimization on the noisy dataset itself.
* **Dual Domain Support:** Specialized handling for both **Static Tabular** data and **Time-Series** (refined globally, with the chain rule summing each time step's contributions across windows).
* **Manifold Preservation:** Achieves state-of-the-art error reduction while maintaining high structural fidelity (minimal $D_{KL}$ and high feature correlation).

---

## 📦 Installation

DenoGrad is available on PyPI and can be installed via pip:

```bash
pip install denograd

```

Alternatively, you can install the latest version from the source:

```bash
git clone https://github.com/JJavier98/DenoGrad.git
cd DenoGrad
pip install .

```

**Requirements:**

* Python >= 3.9
* PyTorch
* NumPy
* tqdm

The installed version is available as `denograd.__version__`. Release notes, including breaking changes, are in the [CHANGELOG](CHANGELOG.md).

---

## 🚀 Quick Start

DenoGrad integrates seamlessly into existing PyTorch pipelines. You simply need your noisy data and a model that has been trained (or partially trained) on it.

### 1. Static Tabular Data Example

```python
import torch
import torch.nn as nn
from denograd import DenoGrad

# 1. Define your model and data
# The model should be pre-trained on the noisy data (or a similar distribution)
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
criterion = nn.MSELoss()

# Assume X_noisy and y_noisy are your numpy arrays
# model.load_state_dict(...) 

# 2. Initialize DenoGrad
denoiser = DenoGrad(model=model, criterion=criterion, device=torch.device('cuda'))

# 3. Fit and Transform
# nrr: Noise Reduction Rate (fraction of the gap to the model closed per epoch)
# nr_threshold: Gating mechanism (don't correct if error < threshold)
X_clean, y_clean, grad_x, grad_y = denoiser.fit_transform(
    X=X_noisy, 
    y=y_noisy,
    nrr=0.5,            
    nr_threshold=0.01,  
    max_epochs=100
)

print("Denoising complete!")

```

### 2. Time-Series Example (Global Refinement)

A time step $t$ appears in every sliding window that covers it. DenoGrad treats the whole series as a single tensor and the windows as views of it, so one backward pass gives each time step the sum of the gradients from all its windows — the chain rule does the consensus.

```python
# 1. Initialize DenoGrad with a recurrent model (e.g., LSTM)
denoiser = DenoGrad(model=lstm_model, criterion=nn.MSELoss())

# 2. Fit and Transform with Time-Series parameters.
# X_ts_noisy has shape (T, n_channels); channel 0 is the series being forecast,
# so it is declared as the target instead of being passed again as y.
X_clean, y_clean, _, _ = denoiser.fit_transform(
    X=X_ts_noisy,
    target_cols=0,       # the target is column 0 of X
    is_ts=True,          # Enable Time-Series mode
    window_size=24,      # Size of the look-back window used by the model
    future=1,            # Steps ahead the model predicts
    lam=1.0,             # weight of the pull towards the observed series
)

```

If the target is not one of the input channels, pass it as `y` instead of `target_cols`.

---

## 🧠 How It Works

Traditional training updates weights ($\theta$) to minimize loss. DenoGrad inverts this process: it freezes $\theta$ and updates the input ($x$). For tabular data each sample takes the step below; time series are refined as a whole (point 4).

$$x_{new} \leftarrow x - \eta \cdot \frac{|r|}{\lVert g \rVert} \cdot \frac{\nabla \mathcal{L}}{\lVert \nabla \mathcal{L} \rVert}$$

The gradient supplies the *direction*, while the step *length* is set by how far the sample sits
from the surface where the model is self-consistent, $\{f_\theta(x) = y\}$. With
$r = f_\theta(x) - y$ the residual and $g = (\partial f_\theta/\partial x, -1)$, that distance is
$|r| / \lVert g \rVert$. A point far from what the network models is corrected hard; one that
already sits close is barely touched. Because $\lVert \nabla \mathcal{L} \rVert \ge \lVert \nabla_y \mathcal{L} \rVert$,
the step can never exceed the residual itself, so no sample can overshoot.

1. **Input Optimization:** The framework calculates the gradient of the loss with respect to the input features and targets.


2. **Gating Mechanism:** To prevent over-smoothing, DenoGrad only updates instances where the prediction error exceeds a user-defined threshold $\tau$ (aleatory margin).


3. **Joint Normalization:** Gradients for features and targets are normalized jointly, and the step is scaled by the sample's own residual (see the formula above) so that no sample moves further than its distance to the consistency surface.


4. **Global Refinement (Time-Series):** A series is refined as a whole rather than window by window. With $z$ every value being refined and $z_{obs}$ its observed value, DenoGrad runs Adam on

   $$\min_z \; \mathcal{L}\big(f_\theta(\text{windows}(z)),\ \text{targets}(z)\big) + \lambda \cdot \operatorname{mean}\big((z - z_{obs})^2\big)$$

   Because the windows are views of one tensor, $\partial \mathcal{L} / \partial z_t$ is the sum of the contributions of every window containing $t$, and a target that is also an input channel receives both of its roles at once. With stride 1 there is one constraint per time step but one unknown per channel and step, so the fit term alone admits many solutions; the proximity term chooses, among the series the model finds self-consistent, the one closest to what was observed.

   The step size no longer depends on each window's residual. Under the previous rule — each window takes its own Gauss-Newton step and a time step receives their average — a backbone that fits well leaves every residual small and barely moves the data. That rule is still available as `ts_strategy="window"`.



---

## 🔧 API Reference

### `DenoGrad` Class

#### `__init__(model, criterion, device=None)`

* `model`: The pre-trained PyTorch model (`nn.Module`).
* `criterion`: The loss function (e.g., `nn.MSELoss`).
* `device`: computing device ('cpu' or 'cuda').

#### `fit_transform(X, y, ...)`

Configures the dataset strategy and executes the denoising loop. It is equivalent to calling `fit(X, y, ...)` followed by `transform(...)`, and accepts the union of their parameters. Use the two separately to fit once and run `transform` several times (e.g. to sweep `nrr` or `lam`): every call starts from the data passed to `fit`, not from the previous call's output.

**Returns:** a tuple `(X_clean, y_clean, grad_x, grad_y)`: the refined inputs and targets as NumPy arrays, and two lists with the per-epoch gradients (empty if `save_gradients=False`; `grad_y` is also empty if `denoise_y=False`, or if the target is given through `target_cols`, since it is then part of `grad_x`).

**General Parameters:**

* `X`, `y`: Input data (Numpy array, Torch Tensor, or Pandas DataFrame). If `X` is a DataFrame, `y` may instead be a column name (or list of names) in `X` to use as the target.
* `nrr` (float, default=0.5): **Noise Reduction Rate**. The fraction of the distance to the model's consistency surface that a sample closes per epoch. It is dimensionless: `nrr=1.0` lands on the surface in a single step, and values between 0.3 and 0.9 converge in a handful of epochs.


* `nr_threshold` (float, default=0.01): **Noise Tolerance**. Corrections are zeroed out if $|y_{pred} - y_{true}| \le \tau$.


* `max_epochs` (int): Maximum number of optimization iterations. Defaults to 100 for the Gauss-Newton step and 300 for global time-series refinement.
* `denoise_y` (bool, default=True): Whether to also refine the target variable.
* `batch_size` (int, default=1000): Number of samples (or windows) processed per gradient computation. Capped at the dataset length.
* `save_gradients` (bool, default=True): Whether to store the per-epoch gradients and return them. Set to `False` to save memory on large datasets.


* `eta_x`, `eta_y` (float, default=1.0): Per-block multipliers on the correction, splitting the step between features and target. Only their **ratio** matters: scaling both by the same factor reaches the same result in more or fewer epochs, exactly as `nrr` does.

  The default `1.0 / 1.0` splits the step as $\lVert \partial f/\partial x \rVert^2 : 1$, which is the maximum-likelihood attribution under isotropic Gaussian noise, and is the right choice for **tabular** data.

  With `ts_strategy="window"` the picture differs. A sequence carries intrinsic stochasticity — the part of the signal the backbone cannot predict from the window — and the refinement removes it along with the measurement noise. Damping the target with `eta_y` between 0.01 and 0.1 limits that loss, and the gain is large: on a series whose backbone reaches $R^2 = 0.94$, the default turns a 33% *increase* in error into a 44% reduction. Sweep `eta_y` before using the window strategy in production.

`nrr`, `nr_threshold`, `eta_x` and `eta_y` belong to the Gauss-Newton step and have no effect on global time-series refinement; `lam`, `lr` and `tol` have no effect on the Gauss-Newton step. Setting one that does not apply raises a warning.

**Time-Series Specific Parameters:**

* `is_ts` (bool): Set to `True` for sequence data.
* `window_size` (int): The input sequence length expected by the model. Required when `is_ts=True`.
* `stride` (int, default=1): Step between the starts of consecutive sliding windows.
* `future` (int): The forecasting horizon (default 1).
* `flattening` (bool): If true, flattens windows (useful for MLP backbones on TS data).
* `target_cols` (int or list of int): Columns of `X` that are the target, as in autoregressive forecasting where the target's history is an input channel. Pass it instead of `y`. Global refinement then treats each target value as the single variable it is, instead of refining an independent copy. Column names passed as `y` with a DataFrame `X` behave the same way. With `denoise_y=False` these columns are held fixed while the other channels move.
* `ts_strategy` (str, default=`"global"`): `"global"` refines the whole series at once (see *How It Works*, point 4). `"window"` is the rule used before 2.0.0 — per-window Gauss-Newton steps averaged over the windows covering each time step — and is kept to reproduce earlier results.
* `lam` (float, default=1.0): Weight of the proximity term under global refinement. Larger values keep the refined series closer to the observed one.
* `lr` (float, default=0.05): Adam learning rate on the data under global refinement.
* `tol` (float, default=1e-7): Global refinement stops once the objective changes by less than this between iterations.

Under global refinement, `batch_size` only bounds memory: the windows are processed in chunks, and the accumulated gradient is the same whatever the chunk size.

---

## 📄 Citation

If you use DenoGrad in your research, please cite our paper:


```bibtex

ON REVISION

```

---

## 👥 Acknowledgments

This work was supported by the **University of Granada** and the **Andalusian Institute of Data Science and Computational Intelligence (DaSCI)**. It is part of the Project "Ethical, Responsible and General Purpose Artificial Intelligence" (IAFER) funded by the European Union Next Generation EU.

---

## 📝 License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) - see the [LICENSE](LICENSE) file for details.
