#!/bin/bash

JOB_NAME="ma_exp"
PARTITION="dgx"

SCRIPTS=(
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/daily_climate/ma.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/ECL/ma.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/ETT/ma.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/microsoft_stock/ma.py"
    "/mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/WTH/ma.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    JOB_SCRIPT="job_script_$(basename $SCRIPT .py).sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_$(basename $SCRIPT .py)
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_$(basename $SCRIPT .py).out
#SBATCH -e /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_$(basename $SCRIPT .py).err

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.

python $SCRIPT
EOF

    chmod +x "$JOB_SCRIPT"
    sbatch "$JOB_SCRIPT"
    rm "$JOB_SCRIPT"
done
