#!/bin/bash
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/daily_climate" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/ECL" --denoising_methods "[\"dae\"]" --device "cuda:1" &\
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/ETT" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/microsoft_stock" --denoising_methods "[\"dae\"]" --device "cuda:1"
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "real/WTH" --denoising_methods "[\"dae\"]" --device "cuda:0" &\
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"dae\"]" --device "cuda:1" &\
python src/scripts/time_series/as_tabular_experiments.py --verbose --data_folder "synthetic/seno" --denoising_methods "[\"dae\"]" --device "cuda:0"