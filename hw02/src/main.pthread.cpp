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
    float xmin = -2.4e-0;
    float xmax =  1.0e-0;
    float ymin = -1.7e-0;
    float ymax =  1.7e-0;
    int    DIM =     500;
    int   save =       1;
    int     nt =       1;
    int   iter =    1000;

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
    }
    // postprocessing
    int xDIM = DIM;
    int yDIM = int(DIM*(ymax-ymin)/(xmax-xmin));

    // print info
    printf("Name: Haoran Sun\n");
    printf("ID:   119010271\n");
    printf("HW:   Mandelbrot Set Computation\n");
    printf("Set xDIM to %d, yDIM to %d\n", xDIM, yDIM);

    // allocation and initialization
    std::complex<float> *Z = (std::complex<float> *)malloc(sizeof(std::complex<float>)*yDIM*xDIM);
    char *map = (char *)malloc(sizeof(char) * xDIM * yDIM);
    mandelbrot_init(Z, xDIM, yDIM, xmin, xmax, ymin, ymax);
    Ptargs *args = (Ptargs *)malloc(sizeof(Ptargs) * nt);
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nt);

    // start time
    auto t1 = std::chrono::system_clock::now();

    // MAIN program
    // create threads
    for (int i = 0; i < nt; i++){
        // calculate start and end index
        int start_idx = xDIM*yDIM/nt * i;
        int end_idx = xDIM*yDIM/nt * (i+1);
        args[i] = (Ptargs){.Z=Z, .map=map, .start_idx=start_idx, .end_idx=end_idx, .iter=iter};
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

    // save png
    if (save==1){
        char filebuff[200];
        snprintf(filebuff, sizeof(filebuff), "mandelbrot_xD%d_yD%d_xR%5.2f-%5.2f_yR%5.2f-%5.2f_iter%d.png",
                 xDIM, yDIM, xmin, xmax, ymin, ymax, iter);
        stbi_write_png(filebuff, xDIM, yDIM, 1, map, 0);
    }

    // free arrays
    free(Z);
    free(map);

    // end time
    printf("Execution time: %.2fs, cpu time: %.2fs, #cpu %2d\n", t, t, 1);

    return 0;
}

