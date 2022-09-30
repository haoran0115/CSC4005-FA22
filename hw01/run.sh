#!/usr/bin/bash
export nproc=1
export arr_size=10000
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

mpirun -np $nproc ./build/bin/main -n $arr_size --save 0

# echo \
# "#!/bin/bash
# #SBATCH --job-name=$jobname
# #SBATCH --nodes=1
# #SBATCH --ntasks=$nproc
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=2048mb
# #SBATCH --partition=Project
# # 

# mpirun -np $nproc ./build/bin/main -n $arr_size
# " > job.sh

# sbatch job.sh

