# DenoGrad: A Model-Agnostic Framework for Gradient-Based Data Refinement

[![PyPI version](https://badge.fury.io/py/denograd.svg)](https://badge.fury.io/py/denograd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DenoGrad** is a novel, model-agnostic framework designed to reduce noise in both input features and target variables by leveraging the gradients of a pre-trained Deep Learning model.

In the Data-Centric AI paradigm, traditional denoising often compromises data integrity by aggressively smoothing features. DenoGrad resolves this by leveraging the **semantic spectral bias** of neural networks. Instead of requiring clean ground truth data, it freezes the weights of your predictive backbone and iteratively backpropagates error corrections into the input space, effectively shifting noisy instances toward the learned data manifold.

### Key Capabilities
* **Model-Agnostic:** Works with any differentiable PyTorch model (MLP, LSTM, CNN, Transformers, TabPFN, etc.).
* **No Clean Ground Truth Required:** Operates via self-supervised input optimization on the noisy dataset itself.
* **Dual Domain Support:** Specialized handling for both **Static Tabular** data and **Time-Series** (via a Consensus Strategy).
* **Manifold Preservation:** Achieves state-of-the-art error reduction while maintaining high structural fidelity (minimal $D_{KL}$ and high feature correlation).

---

## 📦 Installation

DenoGrad is available on PyPI and can be installed via pip:

```bash
pip install denograd

```

Alternatively, you can install the latest version from the source:

```bash
git clone [https://github.com/JJavier98/DenoGrad.git](https://github.com/JJavier98/DenoGrad.git)
cd DenoGrad
pip install -r requirements.txt

```

**Requirements:**

* Python >= 3.8
* PyTorch
* NumPy
* tqdm

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

### 2. Time-Series Example (Consensus Strategy)

For time-series data, DenoGrad employs a **Consensus Strategy**. Since a single time step $t$ appears in multiple sliding windows, DenoGrad accumulates gradients from all contexts and averages them to ensure temporal consistency.

```python
# 1. Initialize DenoGrad with a recurrent model (e.g., LSTM)
denoiser = DenoGrad(model=lstm_model, criterion=criterion)

# 2. Fit and Transform with Time-Series parameters
X_clean, y_clean, _, _ = denoiser.fit_transform(
    X=X_ts_noisy, 
    y=y_ts_noisy,
    is_ts=True,          # Enable Time-Series mode
    window_size=24,      # Size of the look-back window used by the model
    stride=1,
    future=1,            # Steps ahead the model predicts
    nrr=0.5,
    max_epochs=50
)

```

---

## 🧠 How It Works

Traditional training updates weights ($\theta$) to minimize loss. DenoGrad inverts this process: it freezes $\theta$ and updates the input ($x$).

$$x_{new} \leftarrow x - \eta \cdot \frac{|r|}{\lVert g \rVert} \cdot \frac{\nabla \mathcal{L}}{\lVert \nabla \mathcal{L} \rVert}$$

The gradient supplies the *direction*, while the step *length* is set by how far the sample sits
from the surface where the model is self-consistent, $\{f_\theta(x) = y\}$. With
$r = f_\theta(x) - y$ the residual and $g = (\partial f_\theta/\partial x, -1)$, that distance is
$|r| / \lVert g \rVert$. A point far from what the network models is corrected hard; one that
already sits close is barely touched. Because $\lVert \nabla \mathcal{L} \rVert \ge \lVert \nabla_y \mathcal{L} \rVert$,
the step can never exceed the residual itself, so no sample can overshoot.

1. **Input Optimization:** The framework calculates the gradient of the loss with respect to the input features and targets.


2. **Gating Mechanism:** To prevent over-smoothing, DenoGrad only updates instances where the prediction error exceeds a user-defined threshold $\tau$ (aleatory margin).


3. **Joint Normalization:** Gradients for features and targets are normalized jointly to ensure balanced corrections across dimensions.


4. **Consensus Strategy (Time-Series):** For sequential data, gradients are accumulated across all sliding windows covering a time step $t$, and the final update is the average "consensus" direction.



---

## 🔧 API Reference

### `DenoGrad` Class

#### `__init__(model, criterion, device=None)`

* `model`: The pre-trained PyTorch model (`nn.Module`).
* `criterion`: The loss function (e.g., `nn.MSELoss`).
* `device`: computing device ('cpu' or 'cuda').

#### `fit_transform(X, y, ...)`

Configures the dataset strategy and executes the denoising loop.

**General Parameters:**

* `X`, `y`: Input data (Numpy array, Torch Tensor, or Pandas DataFrame).
* `nrr` (float, default=0.5): **Noise Reduction Rate**. The fraction of the distance to the model's consistency surface that a sample closes per epoch. It is dimensionless: `nrr=1.0` lands on the surface in a single step, and values between 0.3 and 0.9 converge in a handful of epochs.


* `nr_threshold` (float, default=0.01): **Noise Tolerance**. Corrections are zeroed out if $|y_{pred} - y_{true}| \le \tau$.


* `max_epochs` (int): Maximum number of optimization iterations.
* `denoise_y` (bool, default=True): Whether to also refine the target variable.


* `eta_x`, `eta_y` (float, default=1.0): Per-block multipliers on the correction, splitting the step between features and target. Only their **ratio** matters: scaling both by the same factor reaches the same result in more or fewer epochs, exactly as `nrr` does.

  The default `1.0 / 1.0` splits the step as $\lVert \partial f/\partial x \rVert^2 : 1$, which is the maximum-likelihood attribution under isotropic Gaussian noise, and is the right choice for **tabular** data.

  For **time series** the picture differs. A sequence carries intrinsic stochasticity — the part of the signal the backbone cannot predict from the window — and the refinement removes it along with the measurement noise. Damping the target with `eta_y` between 0.01 and 0.1 limits that loss, and the gain is large: on a series whose backbone reaches $R^2 = 0.94$, the default turns a 33% *increase* in error into a 44% reduction. Sweep `eta_y` before using time-series mode in production.



**Time-Series Specific Parameters:**

* `is_ts` (bool): Set to `True` for sequence data.
* `window_size` (int): The input sequence length expected by the model.
* `future` (int): The forecasting horizon (default 1).
* `flattening` (bool): If true, flattens windows (useful for MLP backbones on TS data).

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

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```
