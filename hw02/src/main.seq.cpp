#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include <chrono>
#include <thread>
#include "utils.h"


int main(int argc, char* argv[]) {
    // initialization
    float xmin = -2.4e-0;
    float xmax =  1.0e-0;
    float ymin = -1.7e-0;
    float ymax =  1.7e-0;
    int   DIM  =     500;
    int  save  =       1;
    int   iter =    1000;

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

    // start time
    auto t1 = std::chrono::system_clock::now();

    // MAIN program
    mandelbrot_loop(Z, map, 0, xDIM*yDIM, iter);

    // end time
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto dur_ = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
    double t = dur_.count();


    // testing
    // std::complex<float> a(0.1, 0.1);
    // char c = 0;
    // c = mandelbrot_iter(a, a, 100);
    // printf("c = %d\n", c);
    // for (int i = 0; i < xDIM*yDIM; i++){
    //     printf("%f + i%f\n", std::real(Z[i]), std::imag(Z[i]));
    // }
    // printf("\n");

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

