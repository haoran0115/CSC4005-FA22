#!/usr/bin/bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGUI=OFF
cmake --build build
