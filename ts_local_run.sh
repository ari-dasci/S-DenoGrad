#!/bin/bash
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/daily_climate" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/ECL" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/ETT" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/microsoft_stock" --denoising_methods "[\"dlnr\"]" --device "cuda:1" &\
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "real/WTH" --denoising_methods "[\"dlnr\"]" --device "cuda:0" &\
python src/scripts/time_series/ts_experiments.py --verbose --data_folder "synthetic/1000s_5v_24w" --denoising_methods "[\"dlnr\"]" --device "cuda:1"