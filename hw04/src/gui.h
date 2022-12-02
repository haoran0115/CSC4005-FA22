#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <GL/freeglut.h>

int RES = 800;
GLubyte *pix = NULL;

void data2pix(float *temp_arr, GLubyte *pix, int DIM, int RES, float T_bdy, float T_fire){
    for (int i = 0; i < RES; i++){
    for (int j = 0; j < RES; j++){
        int idx_x = i * DIM / RES;
        int idx_y = j * DIM / RES;
        int idx_arr = (idx_x*DIM + idx_y);
        int color = (temp_arr[idx_arr] - T_bdy) * 255 / (T_fire - T_bdy);
        // printf("x = %d, y = %d\n", idx_x, idx_y);
        pix[(i*RES+j)*3] = color;
        pix[(i*RES+j)*3+1] = 255 - color;
        pix[(i*RES+j)*3+2] = 255 - color;
    }}
}


void data2pix_omp(float *temp_arr, GLubyte *pix, int DIM, int RES, float T_bdy, float T_fire){
    #pragma omp parallel for
    for (int i = 0; i < RES; i++){
    for (int j = 0; j < RES; j++){
        int idx_x = i * DIM / RES;
        int idx_y = j * DIM / RES;
        int idx_arr = (idx_x*DIM + idx_y);
        int color = (temp_arr[idx_arr] - T_bdy) * 255 / (T_fire - T_bdy);
        // printf("x = %d, y = %d\n", idx_x, idx_y);
        pix[(i*RES+j)*3] = color;
        pix[(i*RES+j)*3+1] = 255 - color;
        pix[(i*RES+j)*3+2] = 255 - color;
    }}
}
