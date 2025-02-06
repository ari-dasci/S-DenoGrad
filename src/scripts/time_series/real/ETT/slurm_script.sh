#!/bin/bash

JOB_NAME="ETT-grad"
PARTITION="dgx2"

cat > job_script.sh <<EOF
#!/bin/bash
#SBATCH --job-name ${JOB_NAME}
#SBATCH --partition ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_${PARTITION}.out
#SBATCH -e /mnt/homeGPU/JJavierAR/S-noise-gradient/logs/${JOB_NAME}_${PARTITION}.err

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/JJavierAR/repsol_env/
export TFHUB_CACHE_DIR=.

python /mnt/homeGPU/JJavierAR/S-noise-gradient/src/scripts/time_series/real/ETT/gradient.py
EOF

# Dar permisos de ejecución (opcional, pero buena práctica)
chmod +x job_script.sh

# Enviar el script generado a Slurm
sbatch job_script.sh

# (Opcional) Eliminar el script después de enviarlo
rm job_script.sh