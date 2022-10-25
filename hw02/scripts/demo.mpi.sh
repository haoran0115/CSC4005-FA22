#!/usr/bin/bash

# build
./scripts/build.sh 
# mpi
./build/bin/main.pthread -nt 4
