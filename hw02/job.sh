#!/bin/bash
#SBATCH --job-name=proc10
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096mb
#SBATCH --partition=Project
# 
for task in $(seq 500 500 10000); do
    # mpirun -np 10 ./build/bin/main.mpi -n $task --save 0 --record 1
    ./build/bin/main.pthread -nt 10 -n $task --save 0 --record 1
done

