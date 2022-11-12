#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

float *marr_d = NULL;
float *xarr_d = NULL;
float *xarr0_d = NULL;
float *dxarr_d = NULL;

float xmin_d;
float xmax_d;
float ymin_d;
float ymax_d;

int Tx_cu;
int Ty_cu;

