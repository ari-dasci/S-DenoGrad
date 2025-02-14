#!/bin/bash

JOB_NAME="syn_ts"
PARTITION="dgx,dgx2"

SCRIPTS=(
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/dae.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/emd.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/grandient.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/kalman.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/ma.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/pca.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/resnet.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/synthetic/wavelet.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    JOB_SCRIPT="job_script_$(basename $(dirname $SCRIPT)).sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_$(basename $(dirname $SCRIPT))
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_$(basename $(dirname $SCRIPT)).out
#SBATCH -e /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_$(basename $(dirname $SCRIPT)).err

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.

python $SCRIPT
EOF

    chmod +x "$JOB_SCRIPT"
    sbatch "$JOB_SCRIPT"
    rm "$JOB_SCRIPT"
done
