#!/bin/bash

config_prefix="./configs/"

configs=(
    "mtil_order_II"
)

for config in "${configs[@]}"; do
    config_file="${config}.json"
    log_file="${config}.log"
    nohup python ./scripts/run_exp.py \
        --config_path="${config_prefix}${config_file}" \
        > "$log_file" 2>&1 &
done