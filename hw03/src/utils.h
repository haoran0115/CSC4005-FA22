#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>

void print_info(int N, int nsteps){
    printf("Name: Haoran Sun\n");
    printf("ID:   119010271\n");
    printf("HW:   N-Body Simulation\n");
    printf("Set N to %d, nsteps to %d\n", N, nsteps);
}

void partition(int nsteps, int size, int idx, int *start_ptr, int *end_ptr){
    *start_ptr = nsteps / size * idx;
    *end_ptr = nsteps / size * (idx+1);
    if (idx+1==size) *end_ptr = nsteps;
}

void map_idx_to_pair(int N, int idx, int *i_ptr, int *j_ptr){
    int work = N*(N-1) / 2;
    int tmp = (-1 + sqrt(8*idx+9)) / 2;
    int idx_ = tmp * (tmp+1) / 2 - 1;
    if (idx_ < idx) tmp += 1;
    idx_ = tmp * (tmp+1) / 2 - 1;
    *i_ptr = tmp;
    *j_ptr = tmp - 1 + idx - idx_;
    // printf("mmm %d %d\n", *i_ptr, *j_ptr);
}

float norm(float *x, int dim){
    float r = 0;
    for (int i = 0; i < dim; i++){
        r += pow(x[i], 2);
    }
    r = sqrt(r);
    return r;
}

void get_xij(int i, int j, int dim, float *xarr, float *xij, int N){
    for (int k = 0; k < dim; k++){
        xij[k] = xarr[j*dim+k] - xarr[i*dim+k];
    }
}

void print_arr(float *arr, int n){
    for (int i = 0; i < n; i++){
        printf("%10.2f  ", arr[i]);
    }
    printf("\n");
}

void vec_add(float *a, float *b, float *c, 
             float fac1, float fac2, int dim){
    for (int i = 0; i < dim; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}

void vec_add_omp(float *a, float *b, float *c, 
             float fac1, float fac2, int dim){
    #pragma omp parallel for
    for (int i = 0; i < dim; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}

typedef struct pthVecArgs {
    float *a;
    float *b;
    float *c;
    float fac1;
    float fac2;
    int dim;
    int start_idx;
    int end_idx;
} PthVecArgs;


void vec_add_pth_callee(float *a, float *b, float *c, 
    float fac1, float fac2, int dim, int start_idx, int end_idx){
    for (int i = start_idx; i < end_idx; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}

void vec_add_pth_caller(void *vargs){
    PthVecArgs args = *(PthVecArgs *)vargs;
    vec_add_pth_callee(args.a, args.b, args.c, args.fac1, args.fac2, args.dim,
        args.start_idx, args.end_idx);
}


void verlet_at2(int dim, float *marr, float *xarr, float *xarr0,
               float *dxarr, float dt, float G, int N, float cut){
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        // printf("%d %d\n", i, j);
        float xij[dim];
        float tmp[dim];
        float mi = marr[i];
        float mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        float rij = norm(xij, dim);
        float fac = 1.0;
        if (rij < cut) {
            rij = cut;
        }
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/pow(rij, 3);
        }
        // add to dx
        vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, mj*dt*dt, dim);
        vec_add(dxarr+j*dim, dxarr+j*dim, tmp, 1.0, -mi*dt*dt, dim);
    }
}


void verlet_at2_omp(int dim, float *marr, float *xarr, float *xarr0,
               float *dxarr, float dt, float G, int N, float cut){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (j!=i){
            float xij[dim];
            float tmp[dim];
            float mi = marr[i];
            float mj = marr[j];
            // get xij
            get_xij(i, j, dim, xarr, xij, N);
            // compute rij
            float rij = norm(xij, dim);
            float fac = 1.0;
            if (rij < cut) {
                rij = cut;
            }
            // compute intermediate variable
            for (int k = 0; k < dim; k++){
                tmp[k] = xij[k]*G/pow(rij, 3);
            }
            // add to dx
            vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, mj*dt*dt, dim);
            }
        }
    }
}

void verlet_at2_pth_callee(int dim, float *marr, float *xarr, float *xarr0,
    float *dxarr, float dt, float G, int N, float cut,
    int start_idx, int end_idx){
    for (int i = start_idx; i < end_idx; i++){
        for (int j = 0; j < N; j++){
            if (j!=i){
            float xij[dim];
            float tmp[dim];
            float mi = marr[i];
            float mj = marr[j];
            // get xij
            get_xij(i, j, dim, xarr, xij, N);
            // compute rij
            float rij = norm(xij, dim);
            float fac = 1.0;
            if (rij < cut) {
                rij = cut;
            }
            // compute intermediate variable
            for (int k = 0; k < dim; k++){
                tmp[k] = xij[k]*G/pow(rij, 3);
            }
            // add to dx
            vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, mj*dt*dt, dim);
            }
        }
    }
}

typedef struct pthVerArgs{
    int dim;
    float *marr;
    float *xarr;
    float *xarr0;
    float *dxarr;
    float dt;
    float G;
    float N;
    float cut;
    int start_idx;
    int end_idx;
} PthVerArgs;

void verlet_at2_pth_caller(void *vargs){
    PthVerArgs args = *(PthVerArgs *)vargs;
    verlet_at2_pth_callee(args.dim, args.marr, args.xarr, args.xarr0,
        args.dxarr, args.dt, args.G, args.N, args.cut,
        args.start_idx, args.end_idx);
}

void compute_dv(int dim, float *marr, float *xarr, float *dvarr, 
                 float dt, float G, int N, float cut){
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        // printf("%d %d\n", i, j);
        float xij[dim];
        float tmp[dim];
        float mi = marr[i];
        float mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        float rij = norm(xij, dim);
        if (rij < cut) rij = cut;
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/pow(rij, 3);
        }
        // add to dv
        vec_add(dvarr+i*dim, dvarr+i*dim, tmp, 1.0, mj*dt, dim);
        vec_add(dvarr+j*dim, dvarr+j*dim, tmp, 1.0, -mi*dt, dim);
    }
}


void compute_dv_omp(int dim, float *marr, float *xarr, float *dvarr, 
                 float dt, float G, int N, float cut){
    #pragma omp parallel for
    for (int idx = 0; idx < N*(N-1)/2; idx++) {
        int i, j;
        map_idx_to_pair(N, idx, &i, &j);
        float xij[dim];
        float tmp[dim];
        float mi = marr[i];
        float mj = marr[j];
        // get xij
        get_xij(i, j, dim, xarr, xij, N);
        // compute rij
        float rij = norm(xij, dim);
        if (rij < cut) rij = cut;
        // compute intermediate variable
        for (int k = 0; k < dim; k++){
            tmp[k] = xij[k]*G/pow(rij, 3);
        }
        // add to dv
        vec_add(dvarr+i*dim, dvarr+i*dim, tmp, 1.0, mj*dt, dim);
        vec_add(dvarr+j*dim, dvarr+j*dim, tmp, 1.0, -mi*dt, dim);
    }
}


void vec_assign_const(float *a, float c, int dim){
    for (int i = 0; i < dim; i++){
        a[i] = c;
    }
}

void random_generate(float *xarr, float *marr, int N, int dim){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < dim; j++){
            float x = (float) rand() / RAND_MAX * 4 - 2;
            xarr[i*dim+j] = x;
        }
        float m = (float) rand() / RAND_MAX + 1;
        marr[i] = m;
    }
}


void compute_seq(float *xarr, float *xarr0, float *dxarr, int N, int dim, 
    float G, float dt, float radius){
        float *tmp;
        vec_assign_const(dxarr, 0, N*dim);
        verlet_at2(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius); // dx: acc
        vec_add(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);         // dx: x(t)
        vec_add(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);       // dx: x(t-dt)
        tmp = xarr0;
        xarr0 = xarr;
        xarr = tmp;  // switch pointers
        vec_add(xarr, xarr0, dxarr, 1.0, 1.0, N*dim);    // xarr = xarr(0) + dxarr
}

void compute_omp(float *xarr, float *xarr0, float *dxarr, int N, int dim, 
    float G, float dt, float radius){
        float *tmp;
        vec_assign_const(dxarr, 0, N*dim);
        verlet_at2_omp(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius); // dx: acc
        vec_add_omp(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);         // dx: x(t)
        vec_add_omp(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);       // dx: x(t-dt)
        tmp = xarr0;
        xarr0 = xarr;
        xarr = tmp;  // switch pointers
        vec_add_omp(xarr, xarr0, dxarr, 1.0, 1.0, N*dim);    // xarr = xarr(0) + dxarr
}

void compute_pth(float *xarr, float *xarr0, float *dxarr, int N, int dim,
    float G, float dt, float radius, int nt){
    float *tmp;
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nt);
    // call verlet
    for (int i = 0; i < nt; i++){
        int start_idx, end_idx;
        partition(N, nt, i, &start_idx, &end_idx);
        PthVerArgs args = (PthVerArgs){.dim=dim, .marr=marr, .xarr=xarr, .xarr0=xarr0,
            .dxarr=dxarr, .dt=dt, .G=G, .N=N, .cut=radius, .start_idx=start_idx, .end_idx=end_idx};
        // pthread_create(&threads[i], NULL, (void *)&verlet_at2_pth_caller, (void *)(&args));
    }
}
