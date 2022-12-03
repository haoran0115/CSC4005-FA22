#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// global variables
int DIM = 200;    // overall dimension
float T_bdy  = 20; // boundary temperature
float T_fire = 100; // fire temperature
float xmin = -5;
float xmax = 5;
float ymin = -5;
float ymax = 5;
float *temp_arr = NULL;
float *temp_arr0 = NULL;
bool *fire_arr = NULL;
float *x_arr = NULL;
float *y_arr = NULL;

// computing-related constants
int nsteps = 100;

// IO & runtime options
int record = 0;
int nt = 1;

// pthread parameters
pthread_t *thread_arr = NULL;
PthArgs *args_arr = NULL;

// mpi parameters
int size, rank;

// cuda parameters
int Tx = 16;
int Ty = 16;

