#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <iostream>
#include <complex>
#include "stb_image_write.h"

typedef struct ptargs{
    std::complex<float> *Z;
    char *map;
    int start_idx;
    int end_idx;
    int iter;
    int id;
    double *time_arr;
} Ptargs;

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

void mandelbrot_save(const char *jobtype, char *map, 
    int xDIM, int yDIM){
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "mandelbrot_%s.png", jobtype);
    stbi_write_png(filebuff, xDIM, yDIM, 1, map, 0);   
    printf("Image saved as %s.\n", filebuff);
}

void runtime_record(const char *jobtype, int N, int nt, double t, double t_sum){
    FILE* outfile;
    char filebuff[200];
    snprintf(filebuff, sizeof(filebuff), "runtime_%s.txt", jobtype);
    outfile = fopen(filebuff, "a");
    fprintf(outfile, "%10d %5d %10.2f %10.2f\n", N, 1, t, t_sum);
    fclose(outfile);
    printf("Runtime added in %s.\n", filebuff);
}

void runtime_print(int N, int nt, double t, double t_sum){
    printf("Execution time: %.2fs, cpu time: %.2fs, #cpu %2d\n", t, t_sum, nt);
}

