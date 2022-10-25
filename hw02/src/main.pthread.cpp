#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include <chrono>
#include <thread>
#include <pthread.h>
#include "utils.h"


int main(int argc, char* argv[]) {
    // initialization
    float xmin = -2.0e-0;
    float xmax =  0.6e-0;
    float ymin = -1.3e-0;
    float ymax =  1.3e-0;
    int   DIM  =     500;
    int  save  =       1;
    int   iter =     200;
    int record =       0;

    // pthread specific args
    int     nt =       1;

    // parse argument
    char buff[200];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "-n")==0 || strcmp(buff, "--ndim")==0){
            std::string num(argv[i+1]);
            DIM = std::stoi(num);
        }
        if (strcmp(buff, "-nt")==0 || strcmp(buff, "--nthread")==0){
            std::string num(argv[i+1]);
            nt = std::stoi(num);
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

    // print info
    print_info(xDIM, yDIM);

    // allocation and initialization
    std::complex<float> *Z = (std::complex<float> *)malloc(sizeof(std::complex<float>)*yDIM*xDIM);
    char *map = (char *)malloc(sizeof(char) * xDIM * yDIM);
    mandelbrot_init(Z, xDIM, yDIM, xmin, xmax, ymin, ymax);
    Ptargs *args = (Ptargs *)malloc(sizeof(Ptargs) * nt);
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nt);

    // start time
    auto t1 = std::chrono::system_clock::now();
    double *time_arr = (double *)malloc(sizeof(double)*nt);

    // MAIN program
    // create threads
    for (int i = 0; i < nt; i++){
        // calculate start and end index
        int start_idx = xDIM*yDIM/nt * i;
        int end_idx = xDIM*yDIM/nt * (i+1);
        args[i] = (Ptargs){.Z=Z, .map=map, .start_idx=start_idx, .end_idx=end_idx, .iter=iter, .id=i, .time_arr=time_arr};
        if (i==nt-1) args[i].end_idx = xDIM*yDIM;

        // create independent threads
        pthread_create(&threads[i], NULL, mandelbrot_loop_pt, (void *)(&args[i]));
    }
    // join threads
    for (int i = 0; i < nt; i++){
        pthread_join(threads[i], NULL);
    }

    // end time
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto dur_ = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
    double t = dur_.count();
    double t_sum = 0;
    for (int i = 0; i < nt; i++) t_sum += time_arr[i];

    // record data
    if (record==1){
        runtime_record("pt", DIM, nt, t, t_sum);
        runtime_record_detail("pt", DIM, nt, t, time_arr);
    }

    // save png
    if (save==1) mandelbrot_save("pt", map, xDIM, yDIM);

    // end time
    runtime_print(DIM, nt, t, t_sum);

    // rendering
    #ifdef GUI
    // copy memory
    map_glut = (char *)malloc(sizeof(char)*xDIM*yDIM);
    memcpy(map_glut, map, sizeof(char)*xDIM*yDIM);
    // plot
    xDIM_glut = xDIM;
    yDIM_glut = yDIM;
    render("seq");
    free(map_glut);
    #endif


    // free arrays
    free(Z);
    free(map);

    return 0;
}

