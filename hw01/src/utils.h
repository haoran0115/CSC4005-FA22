#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>

// fill random array
void fill_rand_arr(int* arr, int N){
    for (int i = 0; i < N; i++){
        arr[i] = std::rand() % 100;
    }
}

// test binary sort
void odd_even_sort(int* arr, int N, int f){
    if (f==1) return;
    int a, b;
    int flag = 1;
    // odd loop
    for (int i = 1; i < N; i += 2){
        a = arr[i-1];
        b = arr[i];
        if (b < a){
            arr[i]   = a;
            arr[i-1] = b;
            flag = 0;
        }
    }
    // even loop
    for (int i = 2; i < N; i += 2){
        a = arr[i-1];
        b = arr[i];
        if (b < a){
            arr[i]   = a;
            arr[i-1] = b;
            flag = 0;
        }
    }
    return odd_even_sort(arr, N, flag);
}

void print_arr(int* arr, int N){
    for (int i = 0; i < N; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

