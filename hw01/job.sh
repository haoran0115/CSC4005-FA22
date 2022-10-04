#!/bin/bash
#SBATCH --job-name=proc24
#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=2
#SBATCH --mem=2048mb
#SBATCH --partition=Project
# 

mpirun -np 24 ./build/bin/main -n 1000000 --save 1

