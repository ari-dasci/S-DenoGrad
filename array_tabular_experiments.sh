#!/bin/bash
#SBATCH --job-name=tab_exp
#SBATCH --partition=dgx2,dgx
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%6

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

denoising_methods=(
    "dae"
    "dlnr"
    "emd"
    "kalman_filter"
    "moving_average"
    "pca"
    "resnet"
    "wavelet_transform"
)

data_folders=(
    "house_prices"
    "lattice_physics"
    "parkinsons"
    "rt_iot2022"
    "support"
)

# Generate all combinations of denoising_methods and data_folders
combinations=()
for method in "${denoising_methods[@]}"; do
    for folder in "${data_folders[@]}"; do
        combinations+=("$method,$folder")
    done
done

# Get the specific combination for the current SLURM_ARRAY_TASK_ID
IFS=',' read -r method data_folder <<< "${combinations[$SLURM_ARRAY_TASK_ID]}"

output_prefix="logs/tabular/real/${data_folder}/${method}"
mkdir -p "$(dirname "${output_prefix}")"
exec > "${output_prefix}.out" 2> "${output_prefix}.err"

echo "Running experiment with denoising method: $method - on dataset: ${data_folder}"
echo "Running slurm job with id: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python src/scripts/tabular/real/house_prices/all_denoising_methods.py --verbose --data_folder "$data_folder" --denoising_methods "[\"$method\"]" --slurm_id $SLURM_ARRAY_TASK_ID