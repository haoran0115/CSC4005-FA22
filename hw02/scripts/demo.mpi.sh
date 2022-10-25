#!/usr/bin/bash

# build
./scripts/build.sh 
# mpi
mpirun -np 4 ./build/bin/main.mpi
