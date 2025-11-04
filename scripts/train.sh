#!/bin/bash
# run_train.sh

echo "Starting MeanFlow + REPA training..."

accelerate launch train.py --train_config /opt/ai/batik-mean-flow/config.yml

echo "Training complete."