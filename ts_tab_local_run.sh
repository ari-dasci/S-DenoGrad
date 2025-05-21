#!/bin/bash
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/daily_climate" --denoising_methods "[\"dlnr\"]" --device "cuda:0" --noise 0.1 &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/ECL" --denoising_methods "[\"dlnr\"]" --device "cuda:1" --noise 0.1 &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/ETT" --denoising_methods "[\"dlnr\"]" --device "cuda:0" --noise 0.1 &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/microsoft_stock" --denoising_methods "[\"dlnr\"]" --device "cuda:1" --noise 0.1
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/WTH" --denoising_methods "[\"dlnr\"]" --device "cuda:1"
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"resnet\"]" --device "cuda:1"
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"emd\"]" --device "cuda:0" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"kalman_filter\"]" --device "cuda:1" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"moving_average\"]" --device "cuda:0" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"pca\"]" --device "cuda:1" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"wavelet_transform\"]" --device "cuda:0"
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/dwlr" --denoising_methods "[\"dlnr\"]" --device "cuda:0"
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/electric_motor_temperature" --denoising_methods "[\"dlnr\"]" --device "cuda:0" --noise 0.1
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/gas_sensor" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
# python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/pump_sensor" --denoising_methods "[\"dlnr\"]" --device "cuda:0"