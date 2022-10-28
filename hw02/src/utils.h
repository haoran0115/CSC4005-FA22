#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <iostream>
#include <complex>
#include "stb_image_write.h"
#include <sys/stat.h>
#include <sys/types.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>

char *map_glut;
int xDIM_glut, yDIM_glut;
int width = 500;
int xwidth, ywidth;
#endif

typedef struct dsargs{
    int jobsize;
    int curr_idx;
    int max_idx;
    pthread_mutex_t *mutex_ptr;
} Dsargs;

typedef struct ptargs{
    std::complex<float> *Z;
    char *map;
    int start_idx;
    int end_idx;
    int iter;
    int id;
    double *time_arr;
    Dsargs *dsptr;
} Ptargs;

void print_info(int xDIM, int yDIM){
    printf("Name: Haoran Sun\n");
    printf("ID:   119010271\n");
    printf("HW:   Mandelbrot Set Computation\n");
    printf("Set xDIM to %d, yDIM to %d\n", xDIM, yDIM);
}

void mandelbrot_init(std::complex<float> *Z, int xDIM, int yDIM, float xmin, float xmax, float ymin, float ymax){
    for (int i = 0; i < yDIM; i++){
        for (int j = 0; j < xDIM; j++){
            float x = (xmax-xmin)/xDIM*j + xmin;
            float y = (ymin-ymax)/yDIM*i + ymax;
            // printf("%f %f\n", x, y);
            Z[i*xDIM+j] = std::complex<float>(x, y);
        }
    }
}

char mandelbrot_iter(std::complex<float> z, std::complex<float> z0, int iter){
    std::complex<float> p = z;
    for (int i = 0; i < iter; i++){
        z = z * z + z0;
        if (std::real(z * std::conj(z)) > 4) return 255 - 255 * i/iter;
    }
    return 0;
}

void mandelbrot_loop(std::complex<float> *Z, char *map, int start_idx, int end_idx, int iter){
    for (int i = start_idx; i < end_idx; i++){
        map[i] = mandelbrot_iter(Z[i], Z[i], iter);
    }
}

void *mandelbrot_loop_pt(void *vargs){
    // transfer args
    Ptargs args = *(Ptargs *)vargs;
    double *time_arr = args.time_arr;
    int id = args.id;
    // start time
    auto t1 = std::chrono::system_clock::now();

    // main loop
    mandelbrot_loop(args.Z, args.map, args.start_idx, args.end_idx, args.iter);
    
    // end time
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto dur_ = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
    double t = dur_.count();
    time_arr[id] =  t;

    return NULL;
}


void *mandelbrot_loop_pt_ds(void *vargs){
    // transfer args
    Ptargs args = *(Ptargs *)vargs;
    double *time_arr = args.time_arr;
    int id = args.id;
    int max_idx = args.dsptr->max_idx;
    // start time
    auto t1 = std::chrono::system_clock::now();

    // get mutex
    pthread_mutex_t *mutex_ptr = args.dsptr->mutex_ptr;
    while (true){
        // read parameters from global scheduling parameters
        pthread_mutex_lock(mutex_ptr);
        int start_idx = args.dsptr->curr_idx;
        int end_idx   = start_idx + args.dsptr->jobsize;
        if (end_idx > max_idx) end_idx = max_idx;
        args.dsptr->curr_idx = end_idx;
        pthread_mutex_unlock(mutex_ptr);
        if (start_idx>=max_idx) break;
        // main loop
        // printf("id = %d, s = %d, e = %d, m = %d\n", id, start_idx, end_idx, max_idx);
        mandelbrot_loop(args.Z, args.map, start_idx, end_idx, args.iter);
    }
    
    // end time
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto dur_ = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
    double t = dur_.count();
    time_arr[id] =  t;

    return NULL;
}

void mandelbrot_save(const char *jobtype, char *map, 
    int xDIM, int yDIM){
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "mandelbrot_%s.png", jobtype);
    stbi_write_png(filebuff, xDIM, yDIM, 1, map, 0);   
    printf("Image saved as %s.\n", filebuff);
}

void runtime_record(const char *jobtype, int N, int nt, double t, double t_sum){
    const char *folder = "data";
    mkdir(folder, 0777);
    FILE* outfile;
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "./%s/runtime_%s.txt", folder, jobtype);
    outfile = fopen(filebuff, "a");
    fprintf(outfile, "%10d %5d %10.4f %10.4f\n", N, nt, t, t_sum);
    fclose(outfile);
    printf("Runtime added in %s.\n", filebuff);
}

void runtime_record_detail(const char *jobtype, int N, int nt, double t, double *time_arr){
    const char *folder = "data";
    mkdir(folder, 0777);
    FILE* outfile;
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "./%s/runtime_detailed_%s_%d.txt", folder, jobtype, nt);
    outfile = fopen(filebuff, "a");
    fprintf(outfile, "%10d %5d %10.4f ", N, nt, t);
    for (int i = 0; i < nt; i++){
        fprintf(outfile, "%10.4f ", time_arr[i]);
    }
    fprintf(outfile, "\n");
    fclose(outfile);
    printf("Detailed runtime added in %s.\n", filebuff);
}

void runtime_print(int N, int nt, double t, double t_sum){
    printf("Execution time: %.2fs, cpu time: %.2fs, #cpu %2d\n", t, t_sum, nt);
}

#ifdef GUI

void display_test(){
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POLYGON);
    glVertex2f(0, 0);
    glVertex2f(1, 0);
    glVertex2f(1, 1);
    glVertex2f(0, 1);
    glEnd();

    glFlush();
}

void plot(){
    // display test
    // initialization 
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0f, 0.0f, 0.0f);
    
    // draw points
    GLfloat pointSize = 1.0f;
    glPointSize(pointSize);
    glBegin(GL_POINTS);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < yDIM_glut; i++){
            for (int j = 0; j < xDIM_glut; j++){
                int c0 = (unsigned char) map_glut[i*xDIM_glut+j];
                float c = c0;
                c = c0 / 255.0;
                glColor3f(c, c, c);
                glVertex2f(j, yDIM_glut-i);
            }
        }
    glEnd();

    // flush
    glFlush();
}

void resize(int x, int y){
    glutReshapeWindow(xwidth, ywidth);
}


void render(const char *jobtype){
    // glu init
    int glufoo = 1;
    char q[] = " ";
    char *glubar[1];
    glubar[0] = q;
    glutInit(&glufoo, glubar);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    // set x and y width
    xwidth = width;
    ywidth = yDIM_glut*width/xDIM_glut;
    glutInitWindowSize(xwidth, ywidth);
    glutCreateWindow(jobtype);
    glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, xDIM_glut, 0, yDIM_glut);
    
    // display func
    glutDisplayFunc(plot);
    // glutDisplayFunc(display_test);
    glutReshapeFunc(resize);
    
    glutMainLoop();
}

#endif



