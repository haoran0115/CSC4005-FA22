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
    Ptargs args = *(Ptargs *)vargs;
    mandelbrot_loop(args.Z, args.map, args.start_idx, args.end_idx, args.iter);
    return NULL;
}


