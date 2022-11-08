#!/usr/bin/bash
rm -rf tmp
mkdir tmp
nvcc -c -O2 src/cudalib.cu -o tmp/cudalib.o -lcuda
nvcc tmp/cudalib.o src/main.cpp -o tmp/main -I/usr/local/cuda/include -lcuda
./tmp/main
