#!/bin/bash
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"dlnr\"]" --device "cuda:1"

python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"dlnr\"]" --device "cuda:0"