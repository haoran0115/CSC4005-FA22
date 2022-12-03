#include "utils.cuh"
#include "const.cuh"
#define BLOCK_SIZE 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void partition_d(int nsteps, int size, int idx, int *start_ptr, int *end_ptr){
    *start_ptr = nsteps / size * idx;
    *end_ptr = nsteps / size * (idx+1);
    if (idx+1==size) *end_ptr = nsteps;
}

__global__ void print_arr_cu(float *arr, int dim){
    for (int i = 0; i < dim; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

__device__ void print_arr_d(float *arr, int dim){
    for (int i = 0; i < dim; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

void initialize_cu(float *temp_arr, float *temp_arr0, bool *fire_arr,
    float *x_arr, float *y_arr, int DIM, float T_fire, int Tx, int Ty){
    printf("CUDA initialization\n");
    // cuda parameters
    DIM_d = DIM;
    T_fire_d = T_fire;
    Tx_d = Tx;
    Ty_d = Ty;
    // cuda memory allocation
    gpuErrchk( cudaMalloc((void **)&temp_arr_d, sizeof(float)*DIM*DIM) );
    gpuErrchk( cudaMalloc((void **)&temp_arr0_d, sizeof(float)*DIM*DIM) );
    gpuErrchk( cudaMalloc((void **)&fire_arr_d, sizeof(bool)*DIM*DIM) );
    // cuda memory copy
    gpuErrchk( cudaMemcpy(temp_arr_d, temp_arr, sizeof(float)*DIM*DIM, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(temp_arr0_d, temp_arr0, sizeof(float)*DIM*DIM, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(fire_arr_d, fire_arr, sizeof(bool)*DIM*DIM, cudaMemcpyHostToDevice) );
    // synchronize
    cudaDeviceSynchronize();
}

void finalize_cu(){
    printf("CUDA finalization\n");
    // cuda free
    gpuErrchk( cudaFree(temp_arr_d) );
    gpuErrchk( cudaFree(temp_arr0_d) );
    gpuErrchk( cudaFree(fire_arr_d) );
    // synchronize
    cudaDeviceSynchronize();
}

__global__ void update_cu_callee(float *temp_arr, float *temp_arr0, bool *fire_arr,
    float *x_arr, float *y_arr, int DIM, float T_fire){
    int start_idx, end_idx;
    int size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    partition_d(DIM-2, size, idx, &start_idx, &end_idx);
    for (int i = start_idx+1; i < end_idx+1; i++){
    for (int j = 1; j < DIM-1; j++){
        float xw, xa, xs, xd; // w: up; a: left; s: down; d: right
        xw = temp_arr0[i*DIM+j+1];
        xa = temp_arr0[(i-1)*DIM+j];
        xs = temp_arr0[i*DIM+j-1];
        xd = temp_arr0[(i+1)*DIM+j];
        temp_arr[i*DIM+j] = (xw + xa + xs + xd) / 4;
        if (fire_arr[i*DIM+j])
            temp_arr[i*DIM+j] = T_fire;
    }}
}

__global__ void foo(float *arr, int DIM){
    for (int i = 0; i < DIM; i++)
        arr[i] = 0;
}

void update_cu(float *temp_arr){
    update_cu_callee<<<4,4>>>(temp_arr_d, temp_arr0_d, fire_arr_d, 
        NULL, NULL, DIM_d, T_fire_d);
    cudaDeviceSynchronize();

    // copy data to host
    gpuErrchk( cudaMemcpy(temp_arr, temp_arr_d, sizeof(float)*DIM_d*DIM_d, cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();

    // switch pointers
    float *tmp = temp_arr_d;
    temp_arr_d = temp_arr0_d;
    temp_arr0_d = tmp;

    // synchronize
    cudaDeviceSynchronize();
}


