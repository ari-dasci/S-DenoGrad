#!/bin/bash

JOB_NAME="rt_"
PARTITION="dgx2,dgx"

SCRIPTS=(
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/dae.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/emd.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/gradient.py"
    # "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/kalman.py"
    # "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/pca.py"
    # "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/resnet.py"
    # "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/wavelet.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    JOB_SCRIPT="job_script_$(basename $(dirname $SCRIPT)).sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}$(basename $SCRIPT)
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
