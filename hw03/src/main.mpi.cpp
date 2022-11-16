#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <chrono>
#include "const.h"
#include "utils.h"
#ifdef GUI
#include "gui.h"
#endif

void compute(){
    // main program
    char type[] = "mpi";
    int start_idx, end_idx;
    int jobsize = N / size;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    double t;
    partition(N, size, rank, &start_idx, &end_idx);
    if (rank == 0) printf("Start MPI version.\n");
    for (int s = 0; s < nsteps; s++){
        // transfer data
        MPI_Bcast(xarr, N*dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // calculate dx
        vec_assign_const(dxarr, 0, N*dim);
        verlet_at2_part_omp(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius, start_idx, end_idx);
        // verlet_at2_part(dim, marr, xarr, xarr0, dxarr, dt, G, N, radius, start_idx, end_idx);
        vec_add_part(dxarr, dxarr, xarr, 1.0, 1.0, N*dim, start_idx*dim, end_idx*dim);
        vec_add_part(dxarr, dxarr, xarr0, 1.0, -1.0, N*dim, start_idx*dim, end_idx*dim);
        float *tmp = xarr;
        xarr = xarr0;
        xarr0 = tmp;
        MPI_Barrier(MPI_COMM_WORLD);
        verlet_add_part_omp(xarr, xarr0, dxarr, N, dim, xmin, xmax, ymin, ymax, start_idx, end_idx);
        // verlet_add_part(xarr, xarr0, dxarr, N, dim, xmin, xmax, ymin, ymax, start_idx, end_idx);

        // transfer data
        if (rank==0) MPI_Gather(MPI_IN_PLACE, jobsize*dim, MPI_FLOAT, xarr+start_idx*dim, jobsize*dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        else MPI_Gather(xarr+start_idx*dim, jobsize*dim, MPI_FLOAT, xarr, jobsize*dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        // solve tail case
        if (N%jobsize!=0) {
            if (rank==0){
                MPI_Recv(xarr+(N/size*size)*dim, (N%jobsize)*dim, MPI_FLOAT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank+1==size){
                MPI_Send(xarr+(N/size*size)*dim, (N%jobsize)*dim, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // opengl
        if (rank==0){
            #ifdef GUI
            // calculating fps
            int step = 200;
            if (s%step==0 && s%(step*2)!=0) t1 = std::chrono::high_resolution_clock::now();
            else if (s%(step*2)==0 && s!=0) {
                t2 = std::chrono::high_resolution_clock::now();
                t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
                printf("fps: %f frame/s\n", step/t);
            }
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
    }

    // record data
    if (rank==0 && record==1){
        t2 = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t0).count();
        double fps = nsteps / t;
        runtime_record(type, N, size, fps);
    }
}

int main(int argc, char* argv[]){ 
    // mpi initializatio
    MPI_Init(NULL, NULL);
    // fetch size and rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // parse arguments
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
    }

    // print info
    if (rank == 0) print_info(N, nsteps);

    // initialization
    // array allocation
    marr       = (float *)malloc(sizeof(float) * N);
    xarr       = (float *)malloc(sizeof(float) * N * dim);
    xarr0      = (float *)malloc(sizeof(float) * N * dim);
    dxarr      = (float *)malloc(sizeof(float) * N * dim);
    // random generate initial condition
    if (rank == 0){
        random_generate(xarr, marr, N, dim);
        // initialize xarr0
        vec_add(xarr0, xarr0, xarr, 0, 1, N*dim);
    }
    // transfer data
    MPI_Bcast(marr, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(xarr, N*dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(xarr0, N*dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // omp options
    omp_set_dynamic(0);
    omp_set_num_threads(nt);

    // main computing program
    if (rank==0){
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
        #endif
    }
    else {
        compute();
    }

    // mpi finalization
    MPI_Finalize();
}


