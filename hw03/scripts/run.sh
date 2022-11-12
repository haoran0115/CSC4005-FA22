#!/usr/bin/bash
./scripts/build.sh && ./build/bin/main.omp -nt 4 -n 300 --nsteps 100
# ./scripts/build.sh && ./build/bin/main.omp -nt 4 -n 300 --nsteps 100
