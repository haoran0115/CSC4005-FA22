#!/usr/bin/bash
./scripts/build.sh 
paras="--ndim 999 --xmin -0.125 --xmax 0.125 --ymin 0.6 --ymax 0.7"
# ./build/bin/main.seq -n 20 --ndim 1000 --xmin -0.125 --xmax 0.125 --ymin 0.6 --ymax 0.7
# ./build/bin/main.pthread -nt 4 $paras
mpirun -np 4 ./build/bin/main.mpi $paras
