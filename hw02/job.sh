#!/bin/bash
#SBATCH --job-name=proc20
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096mb
#SBATCH --partition=Project
# 
for task in $(seq 500 500 10000); do
    # mpirun -np 20 ./build/bin/main.mpi -n $task --save 0 --record 1
    ./build/bin/main.pthread_ds -nt 20 -n $task --save 0 --record 1
done

