#!/bin/bash

SCRIPTS=(
    "src/scripts/time_series/real/ECL/dae.py"
    "src/scripts/time_series/real/ECL/emd.py"
    "src/scripts/time_series/real/ECL/gradient.py"
    "src/scripts/time_series/real/ECL/kalman.py"
    "src/scripts/time_series/real/ECL/pca.py"
    "src/scripts/time_series/real/ECL/ma.py"
    "src/scripts/time_series/real/ECL/resnet.py"
    "src/scripts/time_series/real/ECL/wavelet.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    python $SCRIPT
done
