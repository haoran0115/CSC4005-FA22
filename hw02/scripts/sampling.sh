#!/usr/bin/bash

repeat=1
cpu_list="$(seq 1 1 10)"
job_start=500
job_incr=500
job_end=10000
p=Project

# repeat=1
# cpu_list="$(seq 1 1 1)"
# job_start=2000
# job_incr=2000
# job_end=2000
# p=Project

task_list=$(seq $job_start $job_incr $job_end)

# sampling parallel data
for m in $(seq $repeat); do
for i in $cpu_list; do
# for k in $task_list; do

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
#SBATCH --mem=4096mb
#SBATCH --partition=$p
# 
for task in \$(seq $job_start $job_incr $job_end); do
    # mpirun -np $i ./build/bin/main.mpi -n \$task --save 0 --record 1
    ./build/bin/main.pthread -nt $i -n \$task --save 0 --record 1
done
" > job.sh

sbatch job.sh
sleep 0.01

# done
done
done

