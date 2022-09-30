#!/bin/bash
#SBATCH --job-name=proc40
#SBATCH --nodes=2
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048mb
#SBATCH --partition=Project
# 

task_list=$(seq 100000 100000 500000)

for k in $task_list; do
    mpirun -np 40 ./build/bin/main -n $k --save 1
done

