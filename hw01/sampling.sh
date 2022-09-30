#!/usr/bin/bash

repeat=1
cpu_list=$(seq 20 10 40)
job_start=100000
job_incr=100000
job_end=500000

for m in $(seq $repeat); do
for i in $cpu_list; do

if [ $i -gt 20 ]; then
    n=2
else
    n=1
fi

echo \
"#!/bin/bash
#SBATCH --job-name=proc$i
#SBATCH --nodes=$n
#SBATCH --ntasks=$i
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048mb
#SBATCH --partition=Project
# 

task_list=\$(seq $job_start $job_incr $job_end)

for k in \$task_list; do
    mpirun -np $i ./build/bin/main -n \$k --save 1
done
" > job.sh

sbatch job.sh

done
done

