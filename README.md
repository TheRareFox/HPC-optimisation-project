# HPC optimisation project

This project benchmarks a building temperature simulation using Jacobi iterations and compares static vs dynamic scheduling with different worker counts.


## Submit commands

Run `bsub < submit.sh`

`bstat` to check job status:


To try out other perimeters you can check the `submit.sh` file

### 1) Run simulation

`python simulate.py 10 --workers 8 --max-iter 20000 --abs-tol 1e-4`

Run dynamic scheduling:

`python simulate.py 10 --workers 8 --dynamic`

Skip plot generation:

`python simulate.py 10 --workers 8 --no-plots`

### 2) Run benchmark (speedup)

Static scheduling benchmark:

`python benchmark_speedup.py --num-buildings 10 --repeats 3 --cores 1,2,3,4,5,6,7,8 --csv output/speedup_static.csv --plot output/speedup_static.png`

Dynamic scheduling benchmark:

`python benchmark_speedup.py --num-buildings 10 --repeats 3 --cores 1,2,3,4,5,6,7,8 --dynamic --csv output/speedup_dynamic.csv --plot output/speedup_dynamic.png`

## Project structure

.

├── benchmark_speedup.py

├── simulate.py

├── submit.sh

├── visualise.ipynb

├── diagrams/

### File overview

- simulate.py: Runs the temperature simulation for one or more buildings.
- benchmark_speedup.py: Repeats simulation runs for multiple worker counts and computes speedup/efficiency.
- submit.sh: LSF batch script for running static and dynamic benchmark jobs on the HPC queue.
- visualise.ipynb: Notebook for exploratory analysis and plotting.
- diagrams/: Diagrams worthy to be saved for documentation.


## Notes

- The scripts expect input data under /dtu/projects/02613_2025/data/modified_swiss_dwellings/.
- If that path is unavailable outside DTU HPC, the simulation will fail at data loading.