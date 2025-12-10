#!/bin/bash


LMDB_PATH="./datasets/AutoLaparo"

LABELS_JSON="./datasets/AutoLaparo.json"

MODEL_DIR="your model"

# Evaluation settings
BATCH_SIZE=32

echo "----------------------------------------------------------------"
echo "Starting Evaluation for AutoLaparo Phase Recognition"
echo "----------------------------------------------------------------"
echo "Data:   $LMDB_PATH"
echo "Labels: $LABELS_JSON"
echo "Models: $MODEL_DIR"
echo "----------------------------------------------------------------"

cd ../downstream

python test_phase_recognition_autolaparo.py \
    --lmdb "$LMDB_PATH" \
    --labels "$LABELS_JSON" \
    --models "$MODEL_DIR" \
    --bs "$BATCH_SIZE" \
