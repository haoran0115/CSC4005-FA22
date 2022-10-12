#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>

// fill random array
void fill_rand_arr(int* arr, int N){
    std::srand(time(0));
    for (int i = 0; i < N; i++){
        arr[i] = std::rand() % 10000;
    }
}

// binary sort
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
// binary sort single iteration

// min-max
void min_max(int *arr, int N) {
    int min_idx = 0;
    int max_idx = 0;
    int min = arr[0];
    int max = arr[0];
    
    for (int i = 1; i < N; i++) {
        if (arr[i] > max) {
            max = arr[i];
            max_idx = i;
        }
        if (arr[i] < min) {
            min = arr[i];
            min_idx = i;
        }
    }

    arr[min_idx] = arr[0];
    arr[max_idx] = arr[N-1];
    arr[0] = min;
    arr[N-1] = max;
}

void print_arr(int* arr, int N){
    for (int i = 0; i < N; i++){
        printf("%d ", arr[i]);
        if ((i+1)%10 == 0) 
            printf("\n");
    }
    if (N%10 != 0) printf("\n");
}

void check_sorted(int* arr, int N){
    for (int i = 0; i < N-1; i++){
        if (arr[i] > arr[i+1]) {
            printf("Error at idx %d.\n", i);
            return;
        }
    }
    printf("Array sorted.\n");
    return;
}


template <class T>
T arr_sum(T *arr, int N){
    T sum = 0;
    for (int i = 0; i < N; i++) sum += arr[i];
    return sum;
}

