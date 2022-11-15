#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>

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

void vec_add_part(float *a, float *b, float *c, 
    float fac1, float fac2, int dim,
    int start_idx, int end_idx){
    for (int i = start_idx; i < end_idx; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
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
        float tmp[dim];
        for (int j = 0; j < dim; j++) tmp[j] = 0;
        for (int j = 0; j < N; j++){
            if (j!=i){
            float xij[dim];
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
                tmp[k] += xij[k]*G/pow(rij, 3) *mj*dt*dt;
            }
            }
        }
        vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, 1.0, dim);
    }
}

void verlet_at2_part(int dim, float *marr, float *xarr, float *xarr0,
    float *dxarr, float dt, float G, int N, float cut,
    int start_idx, int end_idx){
    for (int i = start_idx; i < end_idx; i++){
        float tmp[dim];
        for (int j = 0; j < dim; j++) tmp[j] = 0;
        for (int j = 0; j < N; j++){
            if (j!=i){
            float xij[dim];
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
                tmp[k] += xij[k]*G/pow(rij, 3)*mj*dt*dt;
            }
            }
        }
        vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, 1.0, dim);
    }
}

void verlet_at2_part_omp(int dim, float *marr, float *xarr, float *xarr0,
    float *dxarr, float dt, float G, int N, float cut,
    int start_idx, int end_idx){
    #pragma omp parallel
    {
        int omp_start_idx, omp_end_idx;
        partition(end_idx-start_idx, omp_get_num_threads(), omp_get_thread_num(),
            &omp_start_idx, &omp_end_idx);
        for (int i = start_idx+omp_start_idx; i < start_idx+omp_end_idx; i++){
            float tmp[dim];
            for (int j = 0; j < dim; j++) tmp[j] = 0;
            for (int j = 0; j < N; j++){
                if (j!=i){
                float xij[dim];
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
                    tmp[k] += xij[k]*G/pow(rij, 3)*mj*dt*dt;
                }
                }
            }
            vec_add(dxarr+i*dim, dxarr+i*dim, tmp, 1.0, 1.0, dim);
        }
    }
}

void verlet_add(float *a, float *b, float *c, int N, int dim, 
    int xmin, int xmax, int ymin, int ymax){
    for (int i = 0; i < N; i++){
        float x = b[i*dim+0] + c[i*dim+0];
        float y = b[i*dim+1] + c[i*dim+1];
        if (x < xmin) x += 2 * (xmin - x);
        else if (x > xmax) x += 2 * (xmax - x);
        if (y < ymin) y += 2 * (ymin - y);
        else if (y > ymax) y += 2 * (ymax - y);
        a[i*dim+0] = x;
        a[i*dim+1] = y;
    }
}

void verlet_add_omp(float *a, float *b, float *c, int N, int dim, 
    int xmin, int xmax, int ymin, int ymax){
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        float x = b[i*dim+0] + c[i*dim+0];
        float y = b[i*dim+1] + c[i*dim+1];
        if (x < xmin) x += 2 * (xmin - x);
        else if (x > xmax) x += 2 * (xmax - x);
        if (y < ymin) y += 2 * (ymin - y);
        else if (y > ymax) y += 2 * (ymax - y);
        a[i*dim+0] = x;
        a[i*dim+1] = y;
    }
}

void verlet_add_part(float *a, float *b, float *c, int N, int dim, 
    int xmin, int xmax, int ymin, int ymax,
    int start_idx, int end_idx){
    for (int i = start_idx; i < end_idx; i++){
        float x = b[i*dim+0] + c[i*dim+0];
        float y = b[i*dim+1] + c[i*dim+1];
        if (x < xmin) x += 2 * (xmin - x);
        else if (x > xmax) x += 2 * (xmax - x);
        if (y < ymin) y += 2 * (ymin - y);
        else if (y > ymax) y += 2 * (ymax - y);
        a[i*dim+0] = x;
        a[i*dim+1] = y;
    }
}

void verlet_add_part_omp(float *a, float *b, float *c, int N, int dim, 
    int xmin, int xmax, int ymin, int ymax, int start_idx, int end_idx){
    #pragma omp parallel
    {
        int omp_start_idx, omp_end_idx;
        partition(end_idx-start_idx, omp_get_num_threads(), omp_get_thread_num(),
            &omp_start_idx, &omp_end_idx);
        for (int i = start_idx+omp_start_idx; i < start_idx+omp_end_idx; i++){
            float x = b[i*dim+0] + c[i*dim+0];
            float y = b[i*dim+1] + c[i*dim+1];
            if (x < xmin) x += 2 * (xmin - x);
            else if (x > xmax) x += 2 * (xmax - x);
            if (y < ymin) y += 2 * (ymin - y);
            else if (y > ymax) y += 2 * (ymax - y);
            a[i*dim+0] = x;
            a[i*dim+1] = y;
        }
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


void compute_seq(float **xarr_ptr, float **xarr0_ptr, float *dxarr, float *marr, int N, int dim, 
    float G, float dt, float radius){
    float *tmp;
    float *xarr = *xarr_ptr;
    float *xarr0 = *xarr0_ptr;
    vec_assign_const(dxarr, 0, N*dim);
    verlet_at2(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius); // dx: acc
    vec_add(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);         // dx: x(t)
    vec_add(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);       // dx: x(t-dt)
    *xarr0_ptr = xarr;
    *xarr_ptr = xarr0;  // switch pointers
    xarr = *xarr_ptr;
    xarr0 = *xarr0_ptr;
    verlet_add(xarr, xarr0, dxarr, N, dim, xmin, xmax, ymin, ymax);    // xarr = xarr(0) + dxarr
}

void compute_omp(float **xarr_ptr, float **xarr0_ptr, float *dxarr, float *marr,
    int N, int dim, float G, float dt, float radius){
    float *xarr = *xarr_ptr;
    float *xarr0 = *xarr0_ptr;
    float *tmp;
    vec_assign_const(dxarr, 0, N*dim);
    verlet_at2_omp(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius); // dx: acc
    vec_add_omp(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);         // dx: x(t)
    vec_add_omp(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);       // dx: x(t-dt)
    *xarr0_ptr = xarr;
    *xarr_ptr = xarr0;  // switch pointers
    xarr0 = *xarr0_ptr;
    xarr = *xarr_ptr;
    verlet_add_omp(xarr, xarr0, dxarr, N, dim, xmin, xmax, ymin, ymax);    // xarr = xarr(0) + dxarr
}

typedef struct pthArgs{
    int dim;
    float *marr;
    float *xarr;
    float *xarr0;
    float *dxarr;
    float dt;
    float G;
    int N;
    float cut;
    int nt;
    int idx;
    pthread_barrier_t *barr_ptr;
} PthArgs;

void *compute_pth_callee(void *vargs){
    // initialization
    PthArgs args = *(PthArgs *) vargs;
    int dim = args.dim;
    float *marr = args.marr;
    float *xarr = args.xarr;
    float *xarr0 = args.xarr0;
    float *dxarr = args.dxarr;
    float dt = args.dt;
    float G = args.G;
    int N = args.N;
    float radius = args.cut;
    int nt = args.nt;
    int idx = args.idx;
    pthread_barrier_t *barr_ptr = args.barr_ptr;
    int start_idx, end_idx;

    // verlet algorithm 
    partition(N, nt, idx, &start_idx, &end_idx);
    verlet_at2_part(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius, start_idx, end_idx);
    // vector add
    vec_add_part(dxarr, dxarr, xarr, 1.0, 1.0, N*dim, start_idx*dim, end_idx*dim);
    pthread_barrier_wait(barr_ptr);
    vec_add_part(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim, start_idx*dim, end_idx*dim);
    pthread_barrier_wait(barr_ptr);
    float *tmp = xarr;
    xarr = xarr0;
    xarr0 = tmp;
    pthread_barrier_wait(barr_ptr);
    verlet_add_part(xarr, xarr0, dxarr, N, dim, xmin, xmax, ymin, ymax, start_idx, end_idx);

    return NULL;
}

void compute_pth(float **xarr_ptr, float **xarr0_ptr, float *dxarr, float *marr,
    int N, int dim, float G, float dt, float radius, int nt){
    float *tmp;
    float *xarr = *xarr_ptr;
    float *xarr0 = *xarr0_ptr;
    pthread_t threads[nt];
    pthread_barrier_t barr;
    PthArgs args_arr[nt];
    pthread_barrier_init(&barr, NULL, nt);
    // call verlet
    vec_assign_const(dxarr, 0, N*dim);
    for (int i = 0; i < nt; i++){
        args_arr[i] = (PthArgs){.dim=dim, .marr=marr, .xarr=xarr, .xarr0=xarr0,
            .dxarr=dxarr, .dt=dt, .G=G, .N=N, .cut=radius, 
            .nt=nt, .idx=i, .barr_ptr=&barr};
        pthread_create(&threads[i], NULL, compute_pth_callee, (void *)(&args_arr[i]));
    }
    // join threads
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);
    // switch pointers
    *xarr_ptr = xarr0;
    *xarr0_ptr = xarr;
}

void arr_check_if_identical(float *a, float *b, int dim){
    for (int i = 0; i < dim; i++){
        if (a[i]!=b[i]){
            printf("fuck\n");
            exit(1);
        }
    }
}

void runtime_record(char *jobtype, int N, int nt, double fps){
    const char *folder = "data";
    mkdir(folder, 0777);
    FILE* outfile;
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "./%s/runtime_%s.txt", folder, jobtype);
    outfile = fopen(filebuff, "a");
    fprintf(outfile, "%10d %5d %10.4f\n", N, nt, fps);
    fclose(outfile);
    printf("Runtime added in %s.\n", filebuff);
}
