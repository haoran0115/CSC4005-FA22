#!/usr/bin/bash

# # build the program
# ./scripts/build.sh

# parameters
repeat=1
cpu_list="$(seq 1 1 20)"
job_start=1000
job_incr=1000
job_end=20000
p=Project
task_list=$(seq $job_start $job_incr $job_end)
nsteps=100

# sampling parallel data
for m in $(seq $repeat); do
    for j in $task_list; do
        # # adjust N
        # if [ $j -le 5000 ]; then
        #     nsteps=1000
        # elif [ $j -le 10000 ]; then
        #     nsteps=1000
        # elif [ $j -le 20000 ]; then
        #     nsteps=1000
        # fi
        echo n = $j
        ./build/bin/main.cu --dim $j --record 1 --nsteps $nsteps
    done
done

