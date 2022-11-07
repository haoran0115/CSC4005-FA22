#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

void initialize_cu(double *marr, double *xarr, int N, int dim, int Tx, int Ty);
void compute_cu(double *xarr, int nsteps, int N, int dim, double G, double dt, double cut);
void finalize_cu();

