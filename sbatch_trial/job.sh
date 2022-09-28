#!/bin/bash
#SBATCH --job-name=mpi_helloworld
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=100mb
#SBATCH --partition=Project

mpirun -np 16 ./main

