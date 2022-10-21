#!/usr/bin/bash
./scripts/build.sh 
paras="--ndim 500 --xmin -0.125 --xmax 0.125 --ymin 0.6 --ymax 0.7"
record="--record 1"
# ./build/bin/main.seq $paras
./build/bin/main.pthread -nt 6 $paras $record
# mpirun -np 4 ./build/bin/main.mpi $paras $record
