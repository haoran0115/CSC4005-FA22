#!/usr/bin/bash
cmake -B build -DCMAKE_BUILD_TYPE=Release # -DGUI=ON
cmake --build build -- -j4
