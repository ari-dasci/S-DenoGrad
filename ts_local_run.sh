#!/bin/bash
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/electric_motor_temperature" --denoising_methods "[\"dlnr\"]" --device "cuda:1" --noise 0.1
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/ECL" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/ETT" --denoising_methods "[\"dlnr\"]" --device "cuda:1" --noise 0.15
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/microsoft_stock" --denoising_methods "[\"dlnr\"]" --device "cuda:1" --noise 0.15
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/WTH" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"dlnr\"]" --device "cuda:1"
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"emd\"]" --device "cuda:0" &\
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"kalman_filter\"]" --device "cuda:1"
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"moving_average\"]" --device "cuda:0" &\
# python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"wavelet_transform\"]" --device "cuda:1"