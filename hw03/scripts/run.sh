#!/usr/bin/bash
export OMP_NUM_THREADS=4

./scripts/build.sh
./build/bin/main
