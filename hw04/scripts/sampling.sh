#!/usr/bin/bash

# load mpi modules
module load mpi/mpich-3.2-x86_64

# build the program
./scripts/build.sh
sleep 0.1


# parameters
repeat=1
cpu_list="$(seq 1 1 40)"
job_incr=50
job_start=50
job_end=2000
p=Project
task_list=$(seq $job_start $job_incr $job_end)
nsteps=1000

# sampling parallel data
for m in $(seq $repeat); do
    for j in $task_list; do
        # adjust N
        if [ $j -le 500 ]; then
            nsteps=5000
        elif [ $j -le 1000 ]; then
            nsteps=200
        elif [ $j -le 2000 ]; then
            nsteps=100
        fi
        for i in $cpu_list; do
            # print job info
            echo cpu = $i, n = $j, nsteps = $nsteps

            # sampling
            # ./build/bin/main.omp -nt $i --dim $j --record 1 --nsteps $nsteps
            # ./build/bin/main.pth -nt $i --dim $j --record 1 --nsteps $nsteps
            mpirun -np $i ./build/bin/main.mpi -nt 2 --dim $j --record 1 --nsteps $nsteps
        done
    # ./build/bin/main.seq --dim $j --record 1 --nsteps $nsteps
    done
done

