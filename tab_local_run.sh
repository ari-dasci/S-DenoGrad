#!/bin/bash
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/house_prices" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"


# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"dae\"]" --noise 0.1 --device "cuda:0" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"dlnr\"]" --noise 0.1 --device "cuda:0" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"kalman_filter\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"resnet\"]" --noise 0.1 --device "cuda:1" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"wavelet_transform\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"pca\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"moving_average\"]" --noise 0.1 --device "cpu" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"emd\"]" --noise 0.1 --device "cpu"



# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/lattice_physics" --denoising_methods "[\"dae\"]" --device "cuda:1" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/parkinsons" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/rt_iot2022" --denoising_methods "[\"kalman_filter\"]" --device "cuda:1"
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "real/support" --denoising_methods "[\"dae\"]" --device "cuda:0" &\

# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/2D" --denoising_methods "[\"dae\"]" --device "cuda:1" &\
# python src/scripts/tabular/tabular_experiments.py --verbose --data_folder "synthetic/3D" --denoising_methods "[\"dae\"]" --device "cuda:0"