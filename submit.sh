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

# Run the Python script - Static scheduling
python benchmark_speedup.py --num-buildings 10 --repeats 3 --cores 1,2,4,8 --csv output/speedup_static.csv --plot output/speedup_static.png

# Run the Python script - Dynamic scheduling
python benchmark_speedup.py --num-buildings 10 --repeats 3 --cores 1,2,4,8 --dynamic --csv output/speedup_dynamic.csv --plot output/speedup_dynamic.png