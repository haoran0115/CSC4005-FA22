#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include <chrono>
#include <thread>
#include <mpi.h>
#include "utils.h"


int main(int argc, char* argv[]) {
    // mpi initializatio
    MPI_Init(NULL, NULL);
    // fetch size and rank
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // initialization
    float xmin = -2.0e-0;
    float xmax =  0.6e-0;
    float ymin = -1.3e-0;
    float ymax =  1.3e-0;
    int   DIM  =     500;
    int  save  =       1;
    int   iter =     200;
    int record =       0;

    // parse argument
    char buff[200];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "-n")==0 || strcmp(buff, "--ndim")==0){
            std::string num(argv[i+1]);
            DIM = std::stoi(num);
        }
        if (strcmp(buff, "--xmin")==0){
            std::string num(argv[i+1]);
            xmin = std::stof(num);
        }
        if (strcmp(buff, "--xmax")==0){
            std::string num(argv[i+1]);
            xmax = std::stof(num);
        }
        if (strcmp(buff, "--ymin")==0){
            std::string num(argv[i+1]);
            ymin = std::stof(num);
        }
        if (strcmp(buff, "--ymax")==0){
            std::string num(argv[i+1]);
            ymax = std::stof(num);
        }
        if (strcmp(buff, "--iter")==0){
            std::string num(argv[i+1]);
            iter = std::stof(num);
        }
        if (strcmp(buff, "--save")==0){
            std::string num(argv[i+1]);
            save = std::stoi(num);
        }
        if (strcmp(buff, "--record")==0){
            std::string num(argv[i+1]);
            record = std::stoi(num);
        }
    }
    // postprocessing
    int xDIM = DIM;
    int yDIM = int(DIM*(ymax-ymin)/(xmax-xmin));

    // pre-defined variables
    std::complex<float> *Z;
    std::complex<float> *Z_;
    char *map;
    char *map_;
    int start_idx = xDIM*yDIM/size * rank;
    int end_idx = xDIM*yDIM/size * (rank+1);
    if (rank==size-1) end_idx = xDIM*yDIM;
    // print info
    if (rank==0){
        printf("Name: Haoran Sun\n");
        printf("ID:   119010271\n");
        printf("HW:   Mandelbrot Set Computation\n");
        printf("Set xDIM to %d, yDIM to %d\n", xDIM, yDIM);
        // allocation and initialization
        Z = (std::complex<float> *)malloc(sizeof(std::complex<float>)*yDIM*xDIM);
        map = (char *)malloc(sizeof(char) * xDIM * yDIM);
        mandelbrot_init(Z, xDIM, yDIM, xmin, xmax, ymin, ymax);
    }
    // allocate local variables for each process
    Z_ = (std::complex<float> *)malloc(sizeof(std::complex<float>) * (end_idx-start_idx));
    map_ = (char *)malloc(sizeof(char) * (end_idx-start_idx));

    // timing
    double t1, t2;

    // MAIN program
    // CASE 1: sequential
    if (size==1){
        mandelbrot_loop(Z, map, 0, xDIM*yDIM, iter);
    }
    // CASE 2: parallel
    else {
        // distribute the data
        int scale = sizeof(std::complex<float>) / sizeof(int);
        if (rank==0) {
            for (int i = 1; i < size; i++){
                int start = xDIM*yDIM/size * i;
                int end   = xDIM*yDIM/size * (i+1); if (i==size-1) end = xDIM*yDIM;
                MPI_Send((int *) (Z+start), (end-start)*scale, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            for (int i = 0; i < xDIM*yDIM/size; i++) Z_[i] = Z[i];
        }
        else {
            MPI_Recv((int *) Z_, (end_idx-start_idx)*scale, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // // print check
        // printf("rank %d start_idx %d end_idx %d print %f + %fi\n", 
        //         rank, start_idx, end_idx, std::real(Z_[0]), std::imag(Z_[0]));

        // start timing
        t1 = MPI_Wtime();

        // execution
        mandelbrot_loop(Z_, map_, 0, end_idx-start_idx, iter);

        // end timing
        t2 = MPI_Wtime();

        // gather data
        MPI_Gather(map_, xDIM*yDIM/size, MPI_CHAR, map, xDIM*yDIM/size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        // tail case
        if (xDIM*yDIM%size != 0){
            if (rank == size-1) MPI_Send(map_+xDIM*yDIM/size, xDIM*yDIM%size, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
            if (rank == 0) MPI_Recv(map+xDIM*yDIM/size*size, xDIM*yDIM%size, MPI_CHAR, size-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // end time
    double t = t2 - t1;
    double *time_arr = (double *)malloc(sizeof(double) * size);
    double t_sum = 0;
    MPI_Gather(&t, 1, MPI_DOUBLE, time_arr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i++){
        t_sum += time_arr[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // record data
    if (rank==0 && record==1){
        runtime_record("mpi", xDIM*yDIM, size, t, t_sum);
        runtime_record_detail("mpi", xDIM*yDIM, size, t, time_arr);
    }

    // save png
    if (rank==0 && save==1) mandelbrot_save("mpi", map, xDIM, yDIM);

    // sync
    MPI_Barrier(MPI_COMM_WORLD);

    // end time
    if (rank==0) runtime_print(xDIM*yDIM, size, t, t_sum);

    // rendering
    #ifdef GUI
    if (rank==0){
        // copy memory
        map_glut = (char *)malloc(sizeof(char)*xDIM*yDIM);
        memcpy(map_glut, map, sizeof(char)*xDIM*yDIM);
        // plot
        xDIM_glut = xDIM;
        yDIM_glut = yDIM;
        render("seq");
        free(map_glut);
    }
    #endif

    // free arrays
    if (rank==0){
        free(Z);
        free(map);
    }
    free(Z_);
    free(map_);
    MPI_Barrier(MPI_COMM_WORLD);

    // mpi finalization
    MPI_Finalize();

    return 0;
}

