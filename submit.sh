#!/bin/bash

#BSUB -J python_simulate
#BSUB -q c02613
#BSUB -W 30
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o output/dynamic_core_8.out
#BSUB -e error/dynamic_core_8.err
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"


# Initalise Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run the Python script
# time python -u simulate.py 10 1
# time python -u simulate.py 10 2
# time python -u simulate.py 10 3
# time python -u simulate.py 10 4
# time python -u simulate.py 10 5
# time python -u simulate.py 10 6
# time python -u simulate.py 10 7
time python -u simulate.py 50 8