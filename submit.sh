#!/bin/bash
#BSUB -J python_simulate
#BSUB -q c02613
#BSUB -W 30
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o output/results_all.out
#BSUB -e error/results_all.err
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"

# Initalise Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run the Python script
python -u simulate.py