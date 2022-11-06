#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <omp.h>
#include "const.h"
#include "utils.h"
#include "utils.cuh"
#ifdef GUI
#include "gui.h"
#endif

void compute(){
    // main program
    for (int s = 0; s < nsteps; s++){
        // // step 1: initialize dv and dx to 0 in each step
        // vec_assign_const(dvarr, 0, N*dim);
        // // step 2: compute dv and dx
        // vec_add(dxarr, dxarr, varr, 0.0, dt, N*dim);
        // compute_dv(dim, marr, xarr, dvarr, dt, G, N, radius);
        // // step 3: update v and x
        // vec_add(xarr, xarr, dxarr, 1.0, 1.0, N*dim);
        // vec_add(varr, varr, dvarr, 1.0, 1.0, N*dim);

        // verlet
        vec_assign_const(dxarr, 0, N*dim);
        verlet_at2_omp(dim, marr, xarr, xarr0, dt, G, N, radius);
        vec_add_omp(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);
        vec_add_omp(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);
        vec_add_omp(xarr0, xarr0, xarr, 0.0, 1.0, N*dim);
        vec_add_omp(xarr, xarr, dxarr, 1.0, 1.0, N*dim);

        // // check 
        // print_arr(xarr, N*dim);

        // opengl
        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);

        // gl points
        glBegin(GL_POINTS);
        double xi;
        double yi;
        double xmin, xmax, ymin, ymax;
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
}

int main(int argc, char *argv[]){
    // initialization
    N = 300;
    nsteps = 1e7;
    G = 0.1;
    dt = 0.001;
    marr  = (double *)malloc(sizeof(double) * N);
    xarr  = (double *)malloc(sizeof(double) * N * dim);
    xarr0 = (double *)malloc(sizeof(double) * N * dim);
    varr  = (double *)malloc(sizeof(double) * N * dim);
    dxarr = (double *)malloc(sizeof(double) * N * dim);
    dvarr = (double *)malloc(sizeof(double) * N * dim);

    // random generate initial condition
    random_generate(xarr, marr, N);
    
    // verlet
    vec_add(xarr0, xarr0, xarr, 0, 1, N*dim);

    // main program
    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation Sequential Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glutDisplayFunc(&compute);
    gluOrtho2D(-10, 10, -10, 10);
    glutMainLoop();
    #else
    compute();
    #endif
    // compute();


    return 0;
}
