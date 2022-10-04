#!/usr/bin/bash

repeat=1
# cpu_list="1 $(seq 4 4 40)"
# cpu_list="$(seq 4 4 80)"
cpu_list="24"
# job_start=50000
job_start=600000
job_incr=50000
job_end=1000000
p=Project

task_list=$(seq $job_start $job_incr $job_end)

# sampling parallel data
for m in $(seq $repeat); do
for i in $cpu_list; do
for k in $task_list; do

if [ $i -gt 60 ]; then
    n=4
elif [ $i -gt 40 ]; then
    n=3
elif [ $i -gt 20 ]; then
    n=2
else
    n=1
fi

if [ $i -le 8 ]; then
    p=Debug
    per=1
else 
    p=Project
    per=2
fi

ii=$((i * 2))

echo \
"#!/bin/bash
#SBATCH --job-name=proc$i
#SBATCH --nodes=$n
#SBATCH --ntasks=$i
#SBATCH --cpus-per-task=$per
#SBATCH --mem=2048mb
#SBATCH --partition=$p
# 

mpirun -np $i ./build/bin/main -n $k --save 1
" > job.sh

sbatch job.sh
sleep 0.01

done
done
done

