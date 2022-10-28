#!/usr/bin/bash
./scripts/build.sh 
paras="--ndim 4000 --xmin -0.125 --xmax 0.125 --ymin 0.6 --ymax 0.7"
# paras="--ndim 1000"
record="--record 0"
# ./build/bin/main.seq $paras
# ./build/bin/main.pthread_ds -nt 4 $paras $record
./build/bin/main.pthread -nt 4 $paras $record
# mpirun -np 4 ./build/bin/main.mpi $paras $record
