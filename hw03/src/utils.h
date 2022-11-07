#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>

void partition(int nsteps, int size, int idx, int *start_ptr, int *end_ptr){
    *start_ptr = nsteps / size * idx;
    *end_ptr = nsteps / size * (idx+1);
    if (idx+1==size) *end_ptr = nsteps;
}

void map_idx_to_pair(int N, int idx, int *i_ptr, int *j_ptr){
    int work = N*(N-1) / 2;
    int tmp = (-1 + std::sqrt(8*idx+9)) / 2;
    int idx_ = tmp * (tmp+1) / 2 - 1;
    if (idx_ < idx) tmp += 1;
    idx_ = tmp * (tmp+1) / 2 - 1;
    *i_ptr = tmp;
    *j_ptr = tmp - 1 + idx - idx_;
    // printf("mmm %d %d\n", *i_ptr, *j_ptr);
}

double norm(double *x, int dim){
    double r = 0;
    for (int i = 0; i < dim; i++){
        r += std::pow(x[i], 2);
    }
    r = std::sqrt(r);
    return r;
}

void get_xij(int i, int j, int dim, double *xarr, double *xij, int N){
    for (int k = 0; k < dim; k++){
        xij[k] = xarr[j*dim+k] - xarr[i*dim+k];
    }
}

void print_arr(double *arr, int n){
    for (int i = 0; i < n; i++){
        printf("%10.2f  ", arr[i]);
    }
    printf("\n");
}

void vec_add(double *a, double *b, double *c, 
             double fac1, double fac2, int dim){
    for (int i = 0; i < dim; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}

void vec_add_omp(double *a, double *b, double *c, 
             double fac1, double fac2, int dim){
    #pragma omp parallel for
    for (int i = 0; i < dim; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}

void verlet_at2(int dim, double *marr, double *xarr, double *xarr0,
               double *dxarr, double dt, double G, int N, double cut){
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        // printf("%d %d\n", i, j);
        double xij[dim];
        double tmp[dim];
        double mi = marr[i];
        double mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        double rij = norm(xij, dim);
        double fac = 1.0;
        if (rij < cut) {
            rij = cut;
        }
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/std::pow(rij, 3);
        }
        // add to dx
        vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, mj*dt*dt, dim);
        vec_add(dxarr+j*dim, dxarr+j*dim, tmp, 1.0, -mi*dt*dt, dim);
    }
}

void verlet_at2_omp(int dim, double *marr, double *xarr, double *xarr0,
               double *dxarr, double dt, double G, int N, double cut){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (j!=i){
            double xij[dim];
            double tmp[dim];
            double mi = marr[i];
            double mj = marr[j];
            // get xij
            get_xij(i, j, dim, xarr, xij, N);
            // compute rij
            double rij = norm(xij, dim);
            double fac = 1.0;
            if (rij < cut) {
                rij = cut;
            }
            // compute intermediate variable
            for (int k = 0; k < dim; k++){
                tmp[k] = xij[k]*G/std::pow(rij, 3);
            }
            // add to dx
            vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, mj*dt*dt, dim);
            }
        }
    }
}


void compute_dv(int dim, double *marr, double *xarr, double *dvarr, 
                 double dt, double G, int N, double cut){
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        // printf("%d %d\n", i, j);
        double xij[dim];
        double tmp[dim];
        double mi = marr[i];
        double mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        double rij = norm(xij, dim);
        if (rij < cut) rij = cut;
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/std::pow(rij, 3);
        }
        // add to dv
        vec_add(dvarr+i*dim, dvarr+i*dim, tmp, 1.0, mj*dt, dim);
        vec_add(dvarr+j*dim, dvarr+j*dim, tmp, 1.0, -mi*dt, dim);
    }
}


void compute_dv_omp(int dim, double *marr, double *xarr, double *dvarr, 
                 double dt, double G, int N, double cut){
    #pragma omp parallel for
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        double xij[dim];
        double tmp[dim];
        double mi = marr[i];
        double mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        double rij = norm(xij, dim);
        if (rij < cut) rij = cut;
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/std::pow(rij, 3);
        }
        // add to dv
        vec_add(dvarr+i*dim, dvarr+i*dim, tmp, 1.0, mj*dt, dim);
        vec_add(dvarr+j*dim, dvarr+j*dim, tmp, 1.0, -mi*dt, dim);
    }
}

void vec_assign_const(double *a, double c, int dim){
    for (int i = 0; i < dim; i++){
        a[i] = c;
    }
}

void random_generate(double *xarr, double *marr, int N, int dim){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < dim; j++){
            double x = (double) rand() / RAND_MAX * 4 - 2;
            xarr[i*dim+j] = x;
        }
        double m = (double) rand() / RAND_MAX + 1;
        marr[i] = m;
    }
}

