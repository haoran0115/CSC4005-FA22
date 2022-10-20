#!/usr/bin/bash
./scripts/build.sh 
echo === SEQ ===
# ./build/bin/main.seq -n 20 --print 1
echo ===========
echo
echo === PAR ===
# mpirun -np 4 ./build/bin/main -n 20 --print 1
echo ===========
