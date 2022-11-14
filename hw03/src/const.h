#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// global variables
// computing-related constants
int         N = 30;   // number of particles
int   nsteps = 1e5;   // number of steps
int        dim = 2;   // dimension
float radius = 0.01;   // gravity cut-off
float        G = 0.1;   // gravity constant
float   dt = 0.001;   // time step
float        *marr;   // mass array
float        *xarr;   // position array at time t
float       *xarr0;   // position array at time t - dt
float        *varr;   // velocity array
float       *dxarr;   // position shift array
float       *dvarr;   // velocity shift array
float xmin = -10;
float xmax =  10;
float ymin = -10;
float ymax =  10;

// IO & runtime options
int record = 0;
int nt = 1;

// mpi parameters
int size, rank;
float *xarr_copy;

// cuda parameters
int Tx = 1;
int Ty = 1;

