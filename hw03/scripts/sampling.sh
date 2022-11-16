#!/usr/bin/bash

# build the program
./scripts/build.sh

# parameters
repeat=1
cpu_list="$(seq 1 1 40)"
job_start=50
job_incr=50
job_end=1000
p=Project
task_list=$(seq $job_start $job_incr $job_end)
nsteps=1000

# sampling parallel data
for m in $(seq $repeat); do
    for j in $task_list; do
        # adjust N
        if [ $j -le 100 ]; then
            nsteps=1000
        elif [ $j -le 500 ]; then
            nsteps=200
        elif [ $j -le 1000 ]; then
            nsteps=100
        fi
        for i in $cpu_list; do
            # print job info
            echo cpu = $i, n = $j, nsteps = $nsteps

            # sampling
            ./build/bin/main.omp -nt $i -n $j --record 1 --nsteps $nsteps
            ./build/bin/main.pth -nt $i -n $j --record 1 --nsteps $nsteps
            mpirun -np $i ./build/bin/main.mpi -n $j --record 1 --nsteps $nsteps
        done
    ./build/bin/main.seq -n $j --record 1 --nsteps $nsteps
    done
done

