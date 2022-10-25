#!/bin/bash
#SBATCH --job-name=proc5
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096mb
#SBATCH --partition=Debug
# 
for task in $(seq 500 500 2000); do
    mpirun -np 5 ./build/bin/main.mpi -n $task --save 0 --record 1
done

