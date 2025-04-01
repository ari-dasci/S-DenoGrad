#!/bin/bash

JOB_NAME=""
PARTITION="dgx2,dgx"

SCRIPTS=(
    "src/scripts/tabular/real/house_prices/new_gradient.py"
    # "src/scripts/time_series/real/daily_climate/gradient_more_future.py"
    # "src/scripts/time_series/real/daily_climate/gradient.py"
    # "src/scripts/time_series/real/ECL/gradient.py"
    # "src/scripts/time_series/real/ETT/gradient.py"
    # "src/scripts/time_series/real/microsoft_stock/gradient.py"
    # "src/scripts/time_series/real/WTH/gradient.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    JOB_SCRIPT="job_script_$(basename $(dirname $SCRIPT)).sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}/$(basename $(dirname $SCRIPT))/$(basename $SCRIPT)
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH -o logs/${JOB_NAME}_$(basename $(dirname $SCRIPT)).out
#SBATCH -e logs/${JOB_NAME}_$(basename $(dirname $SCRIPT)).err
#SBATCH -c 16

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.
export PYTHONPATH=$(pwd)/src:$PYTHONPATH  # Añadir src al PYTHONPATH

python $SCRIPT
EOF

    chmod +x "$JOB_SCRIPT"
    sbatch "$JOB_SCRIPT"
    rm "$JOB_SCRIPT"
done
