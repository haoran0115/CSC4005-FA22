#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// global variables
int    N = 2;    // number of particles
int nsteps = 1e5; // number of steps
int skip = 1; // stpes per frame
int dim = 2;  // dimension
double radius = 0.1; // minimum radius
double G = 1;    // gravity constant
double dt = 0.001; // time step
double *marr;    // mass array
double *xarr;    // position array
double *varr;    // velocity array
double *dxarr;   // position shift array
double *dvarr;   // velocity shift array

double *xarr0;

