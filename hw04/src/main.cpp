#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <chrono>
#include "utils.h"
#include "utils.cuh"
#ifdef GUI
#include "gui.h"
#endif
#include "const.h"

void compute(){
    // running type buffer
    char type[1000];
    // start timing
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    double t;
    for (int s = 0; s < nsteps; s++){
        // main compute program
        #ifdef OMP
        update_omp(&temp_arr, &temp_arr0, x_arr, y_arr, DIM, T_fire); 
        #else
        update_seq(&temp_arr, &temp_arr0, x_arr, y_arr, DIM, T_fire); 
        #endif

        // calculating fps
        int step = 30;
        if (s%step==0 && s%(step*2)!=0) t1 = std::chrono::high_resolution_clock::now();
        else if (s%(step*2)==0 && s!=0) {
            t2 = std::chrono::high_resolution_clock::now();
            t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            printf("fps: %f frame/s\n", step/t);
        }
        
        #ifdef GUI
        #ifdef OMP
        data2pix_omp(temp_arr0, pix, DIM, RES, T_bdy, T_fire);
        #else
        data2pix(temp_arr0, pix, DIM, RES, T_bdy, T_fire);
        #endif
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(RES, RES, GL_RGB, GL_UNSIGNED_BYTE, pix);
        glFlush();
        glutSwapBuffers();
        #endif
    }
}

int main(int argc, char *argv[]){
    // parse argument
    char buff[200];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "--dim")==0){
            std::string num(argv[i+1]);
            DIM = std::stoi(num);
        }
        if (strcmp(buff, "-nt")==0){
            std::string num(argv[i+1]);
            nt = std::stoi(num);
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

    // initialization
    temp_arr = (float *)malloc(sizeof(float)*DIM*DIM);
    temp_arr0 = (float *)malloc(sizeof(float)*DIM*DIM);
    fire_arr = (bool *)malloc(sizeof(bool)*DIM*DIM);
    x_arr   = (float *)malloc(sizeof(float)*DIM);
    y_arr   = (float *)malloc(sizeof(float)*DIM);
    #ifdef GUI
    pix = (GLubyte *)malloc(sizeof(GLubyte)*RES*RES*3);
    #endif

    #ifdef OMP
    omp_set_num_threads(nt);
    #else
    #endif
    // assign mesh
    for (int i = 0; i < DIM; i++){
        x_arr[i] = (xmax-xmin) * i/DIM + xmin;
        y_arr[i] = (ymax-ymin) * i/DIM + ymin;
    }
    // assign temperature
    for (int i = 0; i < DIM; i++){
    for (int j = 0; j < DIM; j++){
        float x = x_arr[i];
        float y = y_arr[j];
        temp_arr[i*DIM+j] = T_bdy;
        fire_arr[i*DIM+j] = false;
        if (is_fire(x, y)){
            temp_arr[i*DIM+j] = T_fire;
            fire_arr[i*DIM+j] = true;
        }
    }}
    memcpy(temp_arr0, temp_arr, sizeof(float)*DIM*DIM);
    
    // // check arr status
    // print_arr(x_arr, DIM);
    // print_arr(y_arr, DIM);
    // print_arr(temp_arr, DIM*DIM);

    // print info
    print_info(DIM, nsteps);

    // main program
    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(RES, RES);
    glutCreateWindow("Heat Distribution");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    // glutDisplayFunc(&compute);
    gluOrtho2D(xmin, xmax, ymin, ymax);
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    // glutMainLoop();
    #endif

    compute();

    // finalization
    free(temp_arr);
    free(temp_arr0);
    free(fire_arr);
    free(x_arr);
    free(y_arr);

    return 0;
}

