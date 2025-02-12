#!/bin/bash

JOB_NAME="Rnet"
PARTITION="dgx2"

SCRIPTS=(
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/house_prices/resnet.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/lattice_physics/resnet.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/parkinson/resnet.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/rt_iot2022/resnet.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/tabular/real/support/resnet.py"
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
