#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

float *temp_arr_d = NULL;
float *temp_arr0_d = NULL;
bool *fire_arr_d = NULL;
int DIM_d;
float T_fire_d;

int Tx_d;
int Ty_d;

