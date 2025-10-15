#!/bin/bash

# Setup script for accent-asr project

conda env create -f environment.yml -y
conda activate accent-asr

# Create directories
mkdir -p data/cache
mkdir -p outputs/plots
mkdir -p checkpoints

echo "Setup complete. Activate with: conda activate accent-asr"

