#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>
#include <mpi.h>
#include "utils.h"


int main(int argc, char* argv[]) {
    // mpi initialize
    MPI_Init(NULL, NULL);

    // fetch size and rank
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // initializiation, N = 10 default
    int N = 10;
    // parse argument
    char buff[100];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "-n")==0){
            std::string num(argv[i+1]);
            N = std::stoi(num);
        }
    }

    // main program
    // determine start and end index
    int *arr;
    int *arr_;
    int jobsize = N / size;
    int start_idx = jobsize * rank;
    int end_idx   = start_idx + jobsize;
    int *rbuf = (int *)malloc(sizeof(int) * size);
    int from, to;
    int flag;
    if (rank == size-1) end_idx = N;

    // master proc array allocation
    if (rank==0){
        printf("Set N to %d.\n", N);
        arr_ = (int *) malloc(sizeof(int) * N);
        fill_rand_arr(arr_, N);
        print_arr(arr_, N);
    }
    arr = (int *) malloc(sizeof(int) * (end_idx-start_idx));

    // CASE 1: sequential
    if (size==1) {
        odd_even_sort(arr_, N, 0);
    }

    // CASE 2: parallel
    else {
        // STEP 1: data transfer master --> slave
        if (rank==0){
            for (int i = 1; i < size; i++){
                int start = i*jobsize;
                int end   = start + jobsize;
                MPI_Request request;
                if (i==size-1) end += N%size;
                MPI_Send(arr_+start, end-start, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            for (int i = 0; i < jobsize; i++) arr[i] = arr_[i];
        }
        else
            MPI_Recv(arr, end_idx-start_idx, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        print_arr(arr, end_idx-start_idx);
        
        // STEP 2: main program
        while (true){
            flag = 1;

            // STEP 2.1: local sequential sort
            odd_even_sort(arr, end_idx-start_idx, 0);

            // STEP 2.2: odd-1  <-- odd
            if (rank%2==1) {
                MPI_Send(arr, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
                MPI_Recv(&from, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from > arr[0]) {
                    printf("Exchange rank %d\n", rank);
                    arr[0] = from;
                    odd_even_sort(arr, end_idx-start_idx, MPI_COMM_WORLD);
                    flag = 0;
                }
            }
            else if (rank < size-1) {
                to = arr[end_idx-start_idx-1];
                MPI_Recv(&from, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from < arr[end_idx-start_idx-1]){
                    to = arr[end_idx-start_idx-1];
                    arr[end_idx-start_idx-1] = from;
                    odd_even_sort(arr, end_idx-start_idx, 0);
                    flag = 0;
                }
                MPI_Send(&to, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // STEP 2.2: even-1 <-- even
            if (rank%2==0 && rank>0) {
                MPI_Send(arr, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
                MPI_Recv(&from, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from > arr[0]) {
                    printf("Exchange rank %d\n", rank);
                    arr[0] = from;
                    odd_even_sort(arr, end_idx-start_idx, MPI_COMM_WORLD);
                    flag = 0;
                }
            }
            else if (rank%2==1 && rank<size-1) {
                to = arr[end_idx-start_idx-1];
                MPI_Recv(&from, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from < arr[end_idx-start_idx-1]){
                    to = arr[end_idx-start_idx-1];
                    arr[end_idx-start_idx-1] = from;
                    odd_even_sort(arr, end_idx-start_idx, 0);
                    flag = 0;
                }
                MPI_Send(&to, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // STEP 2.x: sending stop flag to master, master decide whether
            // to continue
            MPI_Gather(&flag, 1, MPI_INT, rbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank==0) {
                print_arr(rbuf, size);
                for (int i = 0; i < size; i++){
                    if (rbuf[i] != 1) {
                        flag = 0;
                    }
                }
            }
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            // printf("2. rank %d flag = %d\n", rank, flag);
            if (flag == 1) break;
        }


        // STEP 3: gather sorted array
        MPI_Gather(arr, jobsize, MPI_INT, arr_, jobsize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        // tail case
        if (N%size != 0) {
            if (rank == size-1) MPI_Send(arr+jobsize, N%size, MPI_INT, 0, 2, MPI_COMM_WORLD);
            if (rank == 0) MPI_Recv(arr_+N/size*size, N%size, MPI_INT, size-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }


    // master print result
    if (rank==0) print_arr(arr_, N);

    // free array 
    free(arr);
    if (rank==0) free(arr_);


    return 0;
}


