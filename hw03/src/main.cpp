#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <chrono>
#include "const.h"
#include "utils.h"
#include "utils.cuh"
#ifdef GUI
#include "gui.h"
#endif

void compute(){
    // running type buffer
    char type[1000];
    // start timing
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    double t;
    // main program
    for (int s = 0; s < nsteps; s++){
        // verlet omp
        #ifdef OMP
        if (s==0) {
            printf("Start OpenMP version.\n");
            strcpy(type, "omp");
        }
        compute_omp(&xarr, &xarr0, dxarr, marr, N, dim, G, dt, radius);
        #elif CUDA
        if (s==0) {
            printf("Start CUDA version.\n");
            strcpy(type, "cuda");
        }
        compute_cu(xarr, nsteps, N, dim, G, dt, radius);
        #elif PTH
        if (s==0) {
            printf("Start Pthread version.\n");
            strcpy(type, "pth");
        }
        compute_pth(&xarr, &xarr0, dxarr, marr, N, dim, G, dt, radius, nt);
        #else
        if (s==0) {
            printf("Start sequential version.\n");
            strcpy(type, "seq");
        }
        compute_seq(&xarr, &xarr0, dxarr, marr, N, dim, G, dt, radius);
        #endif


        // calculating fps
        int step = 200;
        #ifdef GUI
        step = 30;
        #endif
        if (s%step==0 && s%(step*2)!=0) t1 = std::chrono::high_resolution_clock::now();
        else if (s%(step*2)==0 && s!=0) {
            t2 = std::chrono::high_resolution_clock::now();
            t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            printf("fps: %f frame/s\n", step/t);
        }

        // opengl
        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);

        // gl points
        glBegin(GL_POINTS);
        float xi;
        float yi;
        float xmin, xmax, ymin, ymax;
        for (int i = 0; i < N; i++){
            xi = xarr[i*dim+0];
            yi = xarr[i*dim+1];
            glVertex2f(xi, yi);
        }
        glEnd();

        glFlush();
        glutSwapBuffers();
        #endif
    }

    // record data
    if (record==1){
        t2 = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t0).count();
        double fps = nsteps / t;
        runtime_record(type, N, nt, fps);
    }
}

int main(int argc, char *argv[]){
    // parse argument
    char buff[200];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "-n")==0){
            std::string num(argv[i+1]);
            N = std::stoi(num);
        }
        if (strcmp(buff, "-nt")==0){
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
        if (strcmp(buff, "--nsteps")==0){
            std::string num(argv[i+1]);
            nsteps = std::stof(num);
        }
        if (strcmp(buff, "--record")==0){
            std::string num(argv[i+1]);
            record = std::stoi(num);
        }
        if (strcmp(buff, "--Tx")==0){
            std::string num(argv[i+1]);
            Tx = std::stoi(num);
        }
        if (strcmp(buff, "--Ty")==0){
            std::string num(argv[i+1]);
            Ty = std::stoi(num);
        }
    }
    // omp options
    #ifdef OMP
    omp_set_dynamic(0);
    omp_set_num_threads(nt);
    #endif
    
    // print info
    print_info(N, nsteps);

    // array allocation
    marr  = (float *)malloc(sizeof(float) * N);
    xarr  = (float *)malloc(sizeof(float) * N * dim);
    xarr0 = (float *)malloc(sizeof(float) * N * dim);
    dxarr = (float *)malloc(sizeof(float) * N * dim);

    // random generate initial condition
    random_generate(xarr, marr, N, dim);
    print_arr(xarr, 8);
    
    // initialization
    vec_add(xarr0, xarr0, xarr, 0, 1, N*dim);

    // cuda initialize
    #ifdef CUDA
    Tx = 16;
    Ty = 16;
    initialize_cu(marr, xarr, N, dim, Tx, Ty, xmin, xmax, ymin, ymax);
    #endif

    // start timing
    auto t1 = std::chrono::high_resolution_clock::now();
    // main program
    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glutDisplayFunc(&compute);
    glutKeyboardFunc(&guiExit);
    gluOrtho2D(xmin, xmax, ymin, ymax);
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
    #else
    compute();
    // cudaDeviceSynchronize();
    #endif

    // end timing
    auto t2 = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();

    printf("Duration: %fs\n", t);

    // free
    free(marr);
    free(xarr);
    free(xarr0);
    free(dxarr);

    #ifdef CUDA
    // cudafree
    finalize_cu();
    cudaDeviceSynchronize();
    #endif

    return 0;
}
