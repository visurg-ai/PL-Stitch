#!/bin/bash

# ==============================================================================
# LMDB Generation Script for PL-Stitch
# 
# This script converts raw image folders and JSON label files into LMDB format
# for efficient high-speed dataloading.
# ==============================================================================

# 1. Define your base paths
# Change these paths to match your local directory structure
RAW_DATA_ROOT="./datasets/raw"
LMDB_OUTPUT_ROOT="./datasets/lmdb"

# Create output directory if it doesn't exist
mkdir -p "$LMDB_OUTPUT_ROOT"

# ==============================================================================
# 1. PRETRAINING DATASET
# ==============================================================================

# LEMON (Large scale, increase map-size to 2TB or more)
# https://github.com/visurg-ai/LEMON
echo "Creating LMDB for LEMON..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/LEMON/videos" \
    --label-json "$RAW_DATA_ROOT/LEMON/LEMON.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/LEMON_train" \
    --image-size 224 224 \
    --map-size 2e12  # 2TB allocation for large datasets

# ==============================================================================
# 2. EVALUATION DATASETS
# ==============================================================================

# Cholec80
# https://camma.unistra.fr/datasets/
echo "Creating LMDB for Cholec80..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/Cholec80/frames" \
    --label-json "$RAW_DATA_ROOT/Cholec80/Cholec80.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/Cholec80" \
    --image-size 224 224

# AutoLaparo
# https://autolaparo.github.io/
echo "Creating LMDB for AutoLaparo..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/AutoLaparo/frames" \
    --label-json "$RAW_DATA_ROOT/AutoLaparo/AutoLaparo.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/AutoLaparo" \
    --image-size 224 224

# M2CAI16
# https://camma.unistra.fr/datasets/
echo "Creating LMDB for M2CAI16..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/M2CAI16/frames" \
    --label-json "$RAW_DATA_ROOT/M2CAI16/M2CAI16.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/M2CAI16" \
    --image-size 224 224

# Breakfast Actions
# https://serre.lab.brown.edu/breakfast-actions-dataset.html
echo "Creating LMDB for Breakfast..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/Breakfast/videos" \
    --label-json "$RAW_DATA_ROOT/Breakfast/Breakfast.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/Breakfast" \
    --image-size 224 224

# GTEA (Georgia Tech Egocentric Activity)
# https://cbs.ic.gatech.edu/fpv/
echo "Creating LMDB for GTEA..."
python create_lmdb.py \
    --image-folder "$RAW_DATA_ROOT/GTEA/videos" \
    --label-json "$RAW_DATA_ROOT/GTEA/GTEA.json" \
    --lmdb-path "$LMDB_OUTPUT_ROOT/GTEA" \
    --image-size 224 224

echo "All LMDB conversions complete!"
