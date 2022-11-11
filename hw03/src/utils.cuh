#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

void initialize_cu(float *marr, float *xarr, int N, int dim, int Tx, int Ty);
void compute_cu(float *xarr, int nsteps, int N, int dim, float G, float dt, float cut);
void finalize_cu();

