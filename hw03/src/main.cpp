#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <omp.h>
#include "const.h"
#include "utils.h"
#include "utils.cuh"
// #include "const.cuh"
#ifdef GUI
#include "gui.h"
#endif

void compute(){
    // main program
    for (int s = 0; s < nsteps; s++){
        // // verlet omp
        // double *tmp;
        // vec_assign_const(dxarr, 0, N*dim);
        // verlet_at2_omp(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius); // dx: acc
        // vec_add_omp(dxarr, dxarr, xarr, 1.0, 1.0, N*dim);         // dx: x(t)
        // vec_add_omp(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim);       // dx: x(t-dt)
        // tmp = xarr0;
        // xarr0 = xarr;
        // xarr = tmp;  // switch
        // vec_add_omp(xarr, xarr0, dxarr, 1.0, 1.0, N*dim);    // xarr = xarr(0) + dxarr

        // verlet cuda
        // main compute program
        printf("call cuda\n");
        compute_cu(xarr, nsteps, N, dim, G, dt, radius);

        // check 
        print_arr(xarr, N*dim);

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
    N = 5000;
    // N = 3;
    nsteps = 1000;
    // nsteps = 3;
    G = 0.1;
    dt = 0.001;
    marr  = (double *)malloc(sizeof(double) * N);
    xarr  = (double *)malloc(sizeof(double) * N * dim);
    xarr0 = (double *)malloc(sizeof(double) * N * dim);
    dxarr = (double *)malloc(sizeof(double) * N * dim);

    // random generate initial condition
    random_generate(xarr, marr, N, dim);
    // print_arr(xarr, N*dim);
    
    // initialization
    vec_add(xarr0, xarr0, xarr, 0, 1, N*dim);

    // cuda initialize
    Tx = 32;
    Ty = 32;
    initialize_cu(marr, xarr, N, dim, Tx, Ty);

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

    // free
    free(marr);
    free(xarr);
    free(xarr0);
    free(dxarr);

    // cudafree
    finalize_cu();

    return 0;
}
