#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include <mpi.h>
#include <chrono>
#include <thread>
#include "utils.h"


int main(int argc, char* argv[]) {
    // mpi initialize
    MPI_Init(NULL, NULL);

    // fetch size and rank
    int size, rank;
    int save = 0;
    int print = 0;
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
        if (strcmp(buff, "--save")==0){
            std::string num(argv[i+1]);
            save = std::stoi(num);
        }
        if (strcmp(buff, "--print")==0){
            std::string num(argv[i+1]);
            print = std::stoi(num);
        }
    }

    // determine start and end index
    int *arr;
    int *arr_;
    int jobsize = N / size;
    int start_idx = jobsize * rank;
    int end_idx   = start_idx + jobsize;
    int *rbuf = (int *)malloc(sizeof(int) * size);
    double *time_arr = (double *)malloc(sizeof(double) * size);
    double t1, t2, t, t_sum;
    int from, to;
    int flag;
    if (rank == size-1) end_idx = N;

    // master proc array allocation
    if (rank==0){
        printf("Name: Haoran Sun\n");
        printf("ID:   119010271\n");
        printf("HW:   Parallel Odd-Even Sort\n");
        printf("Set N to %d.\n", N);
        arr_ = (int *) malloc(sizeof(int) * N);
        fill_rand_arr(arr_, N);
        if (print==1) {
            printf("Array:\n");
            print_arr(arr_, N);
        }
    }
    arr = (int *) malloc(sizeof(int) * (end_idx-start_idx));

    // MAIN PROGRAM
    // start time
    t1 = MPI_Wtime();

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
        // print_arr(arr, end_idx-start_idx);
        
        // STEP 2: main program
        while (true){
            flag = 1;

            int a, b;
            int from, to;
            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Status status;
            // STEP 2.1: odd loop
            // inner odd loop
            for (int i = 1; i < end_idx-start_idx; i++){
                if ((start_idx+i)%2==1){
                    a = arr[i-1];
                    b = arr[i];
                    if (b < a){
                        arr[i]   = a;
                        arr[i-1] = b;
                        flag = 0;
                    }
                }
            }
            // possible interexchange
            if (start_idx>0 && start_idx%2==1){
                // printf("odd start_idx %d rank %d sendrecv rank %d\n", start_idx, rank, rank-1);
                to = arr[0];
                MPI_Sendrecv(&to, 1, MPI_INT, rank-1, 1, &from, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from > to) {
                    arr[0] = from;
                    flag = 0;
                }
            } 
            else if ((end_idx-1)%2==0 && end_idx<N){
                // printf("odd end_idx %d rank %d sendrecv rank %d\n", end_idx, rank, rank+1);
                to = arr[end_idx-start_idx-1];
                MPI_Sendrecv(&to, 1, MPI_INT, rank+1, 2, &from, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from < to) {
                    arr[end_idx-start_idx-1] = from;
                    flag = 0;
                }
            }
            // MPI_Barrier(MPI_COMM_WORLD);

            // STEP 2.2: even loop
            // inner even loop
            for (int i = 1; i < end_idx-start_idx; i++){
                if ((start_idx+i)%2==0){
                    a = arr[i-1];
                    b = arr[i];
                    if (b < a){
                        arr[i]   = a;
                        arr[i-1] = b;
                        flag = 0;
                    }
                }
            }
            // possible interexchange
            if (rank%2==1 && start_idx>0 && start_idx%2==0){
                // printf("even start_idx %d rank %d sendrecv rank %d\n", start_idx, rank, rank-1);
                to = arr[0];
                MPI_Sendrecv(&to, 1, MPI_INT, rank-1, 1, &from, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from > to) {
                    arr[0] = from;
                    flag = 0;
                }
            } 
            else if (rank%2==0 && (end_idx-1)%2==1 && end_idx<N){
                // printf("even end_idx %d rank %d sendrecv rank %d\n", end_idx, rank, rank+1);
                to = arr[end_idx-start_idx-1];
                MPI_Sendrecv(&to, 1, MPI_INT, rank+1, 2, &from, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from < to) {
                    arr[end_idx-start_idx-1] = from;
                    flag = 0;
                }
            }
            // MPI_Barrier(MPI_COMM_WORLD);
            if (rank%2==0 && start_idx>0 && start_idx%2==0){
                // printf("even start_idx %d rank %d sendrecv rank %d\n", start_idx, rank, rank-1);
                to = arr[0];
                MPI_Sendrecv(&to, 1, MPI_INT, rank-1, 1, &from, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from > to) {
                    arr[0] = from;
                    flag = 0;
                }
            } 
            else if (rank%2==1 && (end_idx-1)%2==1 && end_idx<N){
                // printf("even end_idx %d rank %d sendrecv rank %d\n", end_idx, rank, rank+1);
                to = arr[end_idx-start_idx-1];
                MPI_Sendrecv(&to, 1, MPI_INT, rank+1, 2, &from, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (from < to) {
                    arr[end_idx-start_idx-1] = from;
                    flag = 0;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // STEP 2.3: sending stop flag to master, master decide whether
            // to continue
            MPI_Gather(&flag, 1, MPI_INT, rbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank==0) {
                // print_arr(rbuf, size);
                for (int i = 0; i < size; i++){
                    if (rbuf[i] != 1) {
                        flag = 0;
                    }
                }
            }
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            // printf("2. rank %d flag = %d\n", rank, flag);
            if (flag == 1) {
                // odd_even_sort(arr, end_idx-start_idx, 0);
                break;
            }
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
    if (rank==0 && print==1) {
        printf("Sorted array:\n");
        print_arr(arr_, N);
    }

    // end time
    t2 = MPI_Wtime();
    t = t2 - t1;
    MPI_Gather(&t, 1, MPI_DOUBLE, time_arr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank==0) {
        t_sum = arr_sum(time_arr, size);
        printf("Execution time: %.2fs, cpu time: %.2fs, #cpu %2d\n", t, t_sum, size);
        // check_sorted(arr_, N);
    }

    // free array 
    free(arr);
    if (rank==0) free(arr_);

    // print info to file
    if (rank==0 && save==1) {
        FILE* outfile;
        if (size==1) outfile = fopen("data_seq.txt", "a");
        else  outfile = fopen("data.txt", "a");
        fprintf(outfile, "%10d %5d %10.2f %10.2f\n", N, size, t, t_sum);
        fclose(outfile);
    }
    // this line is added to make sure that the data is correctly saved
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return 0;
}


