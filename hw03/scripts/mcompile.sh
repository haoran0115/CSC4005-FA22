#!/usr/bin/bash
rm -rf tmp
mkdir tmp
nvcc -c -O2 src/cudalib.cu -o tmp/cudalib.o -lcuda
nvcc tmp/cudalib.o src/main.cpp -o tmp/main -I/usr/local/cuda/include -lcuda
./tmp/main
# mpic++
mpic++ src/main.mpi.cpp -lpthread -fopenmp -lGL -lglut -lGLU -DGUI -o build/bin/main.mpi && mpirun -np 2 ./build/bin/main.mpi -nt 20
