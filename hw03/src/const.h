#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// global variables
int    N = 2;    // number of particles
int nsteps = 1e5; // number of steps
int skip = 1; // stpes per frame
int dim = 2;  // dimension
float radius = 0.1; // minimum radius
float G = 1;    // gravity constant
float dt = 0.001; // time step
float *marr;    // mass array
float *xarr;    // position array
float *xarr0;
float *varr;    // velocity array
float *dxarr;   // position shift array
float *dvarr;   // velocity shift array

// cuda parameters
int Tx = 1;
int Ty = 1;

