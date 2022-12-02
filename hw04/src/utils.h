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

void print_info(int DIM, int nsteps){
    printf("Name: Haoran Sun\n");
    printf("ID:   119010271\n");
    printf("HW:   Heat Distribution\n");
    printf("Set DIM to %d, nsteps to %d\n", DIM, nsteps);
}

void partition(int nsteps, int size, int idx, int *start_ptr, int *end_ptr){
    *start_ptr = nsteps / size * idx;
    *end_ptr = nsteps / size * (idx+1);
    if (idx+1==size) *end_ptr = nsteps;
}

void print_arr(float *arr, int n){
    for (int i = 0; i < n; i++){
        printf("%10.2f  ", arr[i]);
    }
    printf("\n");
}

bool is_fire(float x, float y){
    return (x*x + y*y <= 1);
}

void update_seq(float **temp_arr_ptr, float **temp_arr0_ptr, float *x_arr, float *y_arr, int DIM,
    float T_fire){
    float *temp_arr = *temp_arr_ptr;
    float *temp_arr0 = *temp_arr0_ptr;
    for (int i = 1; i < DIM-1; i++){
    for (int j = 1; j < DIM-1; j++){
        float xw, xa, xs, xd; // w: up; a: left; s: down; d: right
        xw = temp_arr0[i*DIM+j+1];
        xa = temp_arr0[(i-1)*DIM+j];
        xs = temp_arr0[i*DIM+j-1];
        xd = temp_arr0[(i+1)*DIM+j];
        temp_arr[i*DIM+j] = (xw + xa + xs + xd) / 4;
        if (is_fire(x_arr[i], y_arr[j])) 
            temp_arr[i*DIM+j] = T_fire;
    }}
    // switch pointers
    *temp_arr_ptr = temp_arr0;
    *temp_arr0_ptr = temp_arr;
}

void update_omp(float **temp_arr_ptr, float **temp_arr0_ptr, float *x_arr, float *y_arr, int DIM,
    float T_fire){
    float *temp_arr = *temp_arr_ptr;
    float *temp_arr0 = *temp_arr0_ptr;
    #pragma omp parallel for
    for (int i = 1; i < DIM-1; i++){
    for (int j = 1; j < DIM-1; j++){
        float xw, xa, xs, xd; // w: up; a: left; s: down; d: right
        xw = temp_arr0[i*DIM+j+1];
        xa = temp_arr0[(i-1)*DIM+j];
        xs = temp_arr0[i*DIM+j-1];
        xd = temp_arr0[(i+1)*DIM+j];
        temp_arr[i*DIM+j] = (xw + xa + xs + xd) / 4;
        if (is_fire(x_arr[i], y_arr[j])) 
            temp_arr[i*DIM+j] = T_fire;
    }}
    // switch pointers
    *temp_arr_ptr = temp_arr0;
    *temp_arr0_ptr = temp_arr;
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
