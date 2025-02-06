#!/bin/bash

#SBATCH --job-name ECL-grad                 # Nombre del proceso
#SBATCH --partition dgx2                    # Cola para ejecutar
#SBATCH --gres=gpu:1 	                    # Numero de gpus a usar
#SBATCH -o /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/ECL_grad_dgx2.out  # Nombre del archivo de salida
#SBATCH -e /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/ECL_grad_dgx2.err  # Nombre del archivo de error
	
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.

# Printing the current working directory
# current_directory=$(pwd)
# echo "Current Working Directory: $current_directory"
python /mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/ECL/gradient.py
