# DenoGrad: Deep Gradient Denoising Framework

[![arXiv](https://img.shields.io/badge/arXiv-2511.10161-b31b1b.svg)](https://arxiv.org/abs/2511.10161)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📄 Description

**DenoGrad** is a novel gradient-based denoising framework designed to enhance the robustness and performance of Artificial Intelligence models, with a specific focus on interpretable (white-box) models.

Unlike conventional techniques that simply remove noisy instances or significantly alter the data distribution, DenoGrad leverages the gradients of a reference Deep Learning (DL) model—trained on the target data—to dynamically detect and correct noisy samples.

### 🚀 Key Features

* **Gradient-Based Correction:** Utilizes gradient information from deep models to guide the noise reduction process effectively.
* **Distribution Preservation:** Corrects instances while maintaining the original data distribution, avoiding oversimplification of the problem space.
* **Task Agnostic:** Validated effectively on both tabular data and time-series datasets.
* **Interpretable AI Enhancement:** Specifically engineered to boost the performance of interpretable models in noisy environments without sacrificing transparency.

## 🛠️ Installation

```bash
pip install denograd
```

## 📖 Basic Usage

```python
from denograd import DenoGrad

# Initialize the denoiser with your reference model
denoiser = DenoGrad(
    model=my_deep_model, # DL model fitted to noisy data
    criterion=nn.MSE(),
    is_ts=False, # Is it a time series problem?
    is_cnn=False # Has the DL model a CNN layer at the beginning?
)

# Fit the noisy data
denoiser.fit(x_noisy, y_noisy)

# Denoise the dataset
x_clean, y_clean, x_gradients, y_gradients = transform(
    nrr = 0.05, # Noise Reduction Rate. Same functionality as learning rate but for denoising porposes.
    nr_threshold = 0.01, # "Level" of noise allowed.
    max_epochs = 200, # Max number of epochs to perform the denoising process.
    plot_progress = False, # 2D and 3D data exclusive.
    path_to_save_imgs = '', # Path where to save the imgs generated for 2D or 3D data.
    denoise_y = True, # It is recommended to set to False for time series (TS) problems.
    batch_size = 1024, # Only used in TS problems.
    save_gradients = True # Save all the gradients calculated through the denoising process.
)
```

## 📝 Citation

If you use DenoGrad in your research, please cite our paper:


> @article{denograd2025,
  title={DenoGrad: Deep Gradient Denoising Framework for Enhancing the Performance of Interpretable AI Models},
  author={Alonso-Ramos, J. Javier and [Other Authors]},
  journal={arXiv preprint arXiv:2511.10161},
  year={2025}
}
