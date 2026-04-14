#!/bin/bash

#BSUB -J python_simulate
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -o output/python_simulate.out 
#BSUB -e error/python_simulate.err
#BSUB -n 8

# Initalise Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run the Python script
time python -u simulate.py 1000 1
# time python -u simulate.py 1000 2
# time python -u simulate.py 1000 3
# time python -u simulate.py 1000 4
# time python -u simulate.py 1000 5
# time python -u simulate.py 1000 6
# time python -u simulate.py 1000 7
# time python -u simulate.py 1000 8