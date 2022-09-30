#!/usr/bin/bash
cmake -B build
cmake --build build
mpirun -np 10 ./build/bin/main -n 31
