#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include <chrono>
#include <thread>
#include "utils.h"


int main(int argc, char* argv[]) {
    // fetch size and rank
    int size = 1, rank = 0;
    int save = 0;
    int print = 0;
    // initializiation, N = 10 default
    int N = 10;
    // parse argument
    char buff[100];
    for (int i = 0; i < argc; i++){
        strcpy(buff, argv[i]);
        if (strcmp(buff, "-n")==0){
            std::string num(argv[i+1]);
            N = std::stoi(num);
        }
        if (strcmp(buff, "--save")==0){
            std::string num(argv[i+1]);
            save = std::stoi(num);
        }
        if (strcmp(buff, "--print")==0){
            std::string num(argv[i+1]);
            print = std::stoi(num);
        }
    }

    // determine start and end index
    int *arr = (int *)malloc(sizeof(int) * N);
    int *rbuf = (int *)malloc(sizeof(int) * size);

    // master proc array allocation
    printf("Name: Haoran Sun\n");
    printf("ID:   119010271\n");
    printf("HW:   Parallel Odd-Even Sort\n");
    printf("Set N to %d.\n", N);
    fill_rand_arr(arr, N);
    if (print==1) {
        printf("Array:\n");
        print_arr(arr, N);
    }

    // MAIN PROGRAM
    // start time
    auto t1 = std::chrono::system_clock::now();

    // sequential sort
    odd_even_sort(arr, N, 0);

    // print array
    if (print==1) {
        printf("Sorted array:\n");
        print_arr(arr, N);
    }

    // end time
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto dur_ = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
    double t = dur_.count();
    printf("Execution time: %.2fs, cpu time: %.2fs, #cpu %2d\n", t, t, size);

    // free array 
    free(arr);

    // print data info to file
    if (save==1) {
        FILE* outfile;
        outfile = fopen("data_seq.txt", "a");
        fprintf(outfile, "%10d %5d %10.2f %10.2f\n", N, size, t, t);
        fclose(outfile);
    }

    // this line is added to make sure that the data is correctly saved
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return 0;
}


