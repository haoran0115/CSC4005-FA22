#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>


void initialize_cu(float *temp_arr, float *temp_arr0, bool *fire_arr,
    float *x_arr, float *y_arr, int DIM, float T_fire, int Tx, int Ty);
void finalize_cu();
void update_cu(float *temp_arr);
void copy_cu(float *temp_arr);
