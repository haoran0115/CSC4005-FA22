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

    // print info
    print_info(xDIM, yDIM);

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

    // record data
    if (record==1) runtime_record("seq", DIM, 1, t, t);

    // save png
    if (save==1) mandelbrot_save("seq", map, xDIM, yDIM);

    // end time
    runtime_print(DIM, 1, t, t);

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

