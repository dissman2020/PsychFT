#!/bin/bash

echo "Current directory: $(pwd)"

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/sft_wd.py \
--config recipes/Llama-3.2-1B/sft/config_demo.yaml
