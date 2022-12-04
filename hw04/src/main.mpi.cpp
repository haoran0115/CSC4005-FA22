#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory.h>
#include <chrono>
#include "utils.h"
#ifdef GUI
#include "gui.h"
#endif
#include "const.h"

void compute(){
    // running type buffer
    char type[1000] = "mpi";
    // start timing
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    double t;
    // mpi computing parameters
    int start_idx, end_idx;
    int jobsize = DIM / size;
    partition(DIM, size, rank, &start_idx, &end_idx);
    for (int s = 0; s < nsteps; s++){
        // transfer data
        MPI_Bcast(temp_arr0, DIM*DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // main compute program
        update_omp_part(&temp_arr, &temp_arr0, fire_arr, x_arr, y_arr, DIM, T_fire,
            start_idx, end_idx);
        MPI_Barrier(MPI_COMM_WORLD);

        // transfer data
        if (rank==0) MPI_Gather(MPI_IN_PLACE, jobsize*DIM, MPI_FLOAT, temp_arr0+start_idx*DIM,
            jobsize*DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        else MPI_Gather(temp_arr0+start_idx*DIM, jobsize*DIM, MPI_FLOAT, temp_arr0, jobsize*DIM,
            MPI_FLOAT, 0, MPI_COMM_WORLD);
        // solve tail case
        if (DIM%jobsize!=0) {
            if (rank==0){
                MPI_Recv(temp_arr0+(DIM/size*size)*DIM, (DIM%jobsize)*DIM, MPI_FLOAT, size-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank+1==size){
                MPI_Send(temp_arr0+(DIM/size*size)*DIM, (DIM%jobsize)*DIM, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // calculate fps
        int step = 60;
        if (s%step==0 && s%(step*2)!=0) t1 = std::chrono::high_resolution_clock::now();
        else if (s%(step*2)==0 && s!=0) {
            t2 = std::chrono::high_resolution_clock::now();
            t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
            if (rank==0) printf("fps: %f frame/s\n", step/t);
        }
        
        #ifdef GUI
        if (rank==0){
            data2pix_omp(temp_arr0, pix, DIM, RES, T_bdy, T_fire);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawPixels(RES, RES, GL_RGB, GL_UNSIGNED_BYTE, pix);
            glFlush();
            glutSwapBuffers();
        }
        #endif
    }

    // record data
    if (rank==0 && record==1){
        t2 = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t0).count();
        double fps = nsteps / t;
        runtime_record(type, DIM, size, fps);
    }
}

int main(int argc, char *argv[]){
    // mpi initialize
    MPI_Init(NULL, NULL);
    // fetch size and rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    // omp initialize
    omp_set_num_threads(nt);


    // print info
    if (rank==0) print_info(DIM, nsteps);

    // initialization
    temp_arr = (float *)malloc(sizeof(float)*DIM*DIM);
    temp_arr0 = (float *)malloc(sizeof(float)*DIM*DIM);
    fire_arr = (bool *)malloc(sizeof(bool)*DIM*DIM);
    x_arr   = (float *)malloc(sizeof(float)*DIM);
    y_arr   = (float *)malloc(sizeof(float)*DIM);
    #ifdef GUI
    if (rank==0) pix = (GLubyte *)malloc(sizeof(GLubyte)*RES*RES*3);
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
 
    // main program
    #ifdef GUI
    if (rank==0){
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(RES, RES);
        glutCreateWindow("Heat Distribution");
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        gluOrtho2D(xmin, xmax, ymin, ymax);
        glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    }
    #endif

    compute();

    // finalization
    free(temp_arr);
    free(temp_arr0);
    free(fire_arr);
    free(x_arr);
    free(y_arr);

    #ifdef GUI
    if (rank==0) free(pix);
    #endif

    return 0;
}

