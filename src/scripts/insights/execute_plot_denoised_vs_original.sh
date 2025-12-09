#!/bin/bash

# Base directory for data
DATA_DIR="data"
SCRIPT_PATH="/home/jjavier98/S-noise-gradient/src/scripts/insights/plot_denoised_vs_original.py"

# Iterate over data types (tabular, time_series)
for data_type in "time_series"; do
    data_type_path="$DATA_DIR/$data_type"
    if [ ! -d "$data_type_path" ]; then
        continue
    fi

    # Iterate over data origins (real, synthetic)
    for data_origin in "real" "synthetic"; do
        data_origin_path="$data_type_path/$data_origin"
        if [ ! -d "$data_origin_path" ]; then
            continue
        fi

        # Iterate over datasets
        for dataset_name in "$data_origin_path"/*; do
            if [ ! -d "$dataset_name" ]; then
                continue
            fi

            dataset_name=$(basename "$dataset_name")

            # Iterate over models in the denoised folder
            denoised_path="$data_origin_path/$dataset_name/denoised"
            if [ ! -d "$denoised_path" ]; then
                continue
            fi

            for model_file in "$denoised_path"/*.parquet; do
                model_name=$(basename "$model_file" | sed 's/_denoised\.parquet$//')

                if [ $model_name != "dae" ]; then
                    continue
                fi

                # Execute the Python script
                python "$SCRIPT_PATH" --data_type "$data_type" \
                                       --data_origin "$data_origin" \
                                       --dataset_name "$dataset_name" \
                                       --model_name "$model_name" \
                                    #    --num_points 1000
            done
        done
    done
done