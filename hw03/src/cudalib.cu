#include "utils.cuh"
#include "const.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void get_xij_d(int i, int j, int dim, float *xarr, float *xij, int N){
    for (int k = 0; k < dim; k++){
        xij[k] = xarr[j*dim+k] - xarr[i*dim+k];
    }
}

__device__ void partition_d(int nsteps, int size, int idx, int *start_ptr, int *end_ptr){
    *start_ptr = nsteps / size * idx;
    *end_ptr = nsteps / size * (idx+1);
    if (idx+1==size) *end_ptr = nsteps;
}

__device__ float norm_d(float *x, int dim){
    float r = 0;
    for (int i = 0; i < dim; i++){
        r += x[i]*x[i];
    }
    r = sqrt(r);
    return r;
}

__device__ void vec_add_d(float *a, float *b, float *c, 
    float fac1, float fac2, int dim){
    for (int i = 0; i < dim; i++){
        a[i] = fac1*b[i] + fac2*c[i];
    }
}


__global__ void vec_add_cu(float *a, float *b, float *c, int dim){
    int size = blockDim.x * gridDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int start_idx, end_idx;
    partition_d(dim, size, idx, &start_idx, &end_idx);
    for (int i = start_idx; i < end_idx; i++){
        a[i] = b[i] + c[i];
    }
}

__global__ void vec_sub_cu(float *a, float *b, float *c, int dim){
    int size = blockDim.x * gridDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int start_idx, end_idx;
    partition_d(dim, size, idx, &start_idx, &end_idx);
    for (int i = start_idx; i < end_idx; i++){
        a[i] = b[i] - c[i];
    }
}

__global__ void gather_dx_cu(float *a, float *b, float *c, int dim){
    int size = blockDim.x * gridDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int start_idx, end_idx;
    partition_d(dim, size, idx, &start_idx, &end_idx);
    for (int i = start_idx; i < end_idx; i++){
        a[i] += b[i] - c[i];
    }
}

__global__ void verlet_at2_cu(const int dim, float *marr, float *xarr, float *xarr0,
    float *dxarr, float dt, float G, int N, float cut){
    int size = blockDim.x * gridDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int start_idx, end_idx;
    partition_d(N, size, idx, &start_idx, &end_idx);
    // printf("%d %d\n", start_idx, end_idx);
    // TODO: check later
    for (int i = start_idx; i < end_idx; i++){
        float tmp0 = 0.0;
        float tmp1 = 0.0;
        for (int j = 0; j < N; j++){
            if (j!=i){
            // get xij
            float xij0 = xarr[j*dim+0] - xarr[i*dim+0];
            float xij1 = xarr[j*dim+1] - xarr[i*dim+1];
            // compute rij
            float rij = sqrt(xij0*xij0 + xij1*xij1);
            float fac = 1.0;
            if (rij < cut) {
                rij = cut;
            }
            tmp0 += xij0 * G/(rij*rij*rij) * marr[j]*dt*dt;
            tmp1 += xij1 * G/(rij*rij*rij) * marr[j]*dt*dt;
            }
        }
        dxarr[i*dim + 0] = tmp0;
        dxarr[i*dim + 1] = tmp1;
    }
}

__global__ void print_arr_cu(float *arr, int dim){
    printf("parr_cu1\n");
    for (int i = 0; i < dim; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
    printf("parr_cu2\n");
}

// cuda initialize program
void initialize_cu(float *marr, float *xarr, int N, int dim, int Tx, int Ty){
    printf("cuda initialize\n");
    // cuda parameters
    Tx_cu = Tx;
    Ty_cu = Ty;
    // cuda memory allocation
    gpuErrchk( cudaMalloc((void **) &marr_d, sizeof(float)*N));
    gpuErrchk( cudaMalloc((void **) &xarr_d, sizeof(float)*N*dim));
    gpuErrchk( cudaMalloc((void **) &xarr0_d, sizeof(float)*N*dim));
    gpuErrchk( cudaMalloc((void **) &dxarr_d, sizeof(float)*N*dim));
    // copy
    gpuErrchk( cudaMemcpy(marr_d, marr, sizeof(float)*N, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(xarr_d, xarr, sizeof(float)*N*dim, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(xarr0_d, xarr, sizeof(float)*N*dim, cudaMemcpyHostToDevice) );
    
    // print check: passed
    // print_arr_cu<<<1,1>>>(marr_d, N);    
    cudaDeviceSynchronize();
}

// switch pointers
__global__ void swap(float * &a, float * &b){
    float *tmp = a;
    a = b;
    b = tmp;
}

// verlet cuda callee
void compute_cu(float *xarr, int nsteps, int N, int dim, float G, float dt, float cut){
    // verlet cuda main program
    float *tmp;
    cudaMemset(dxarr_d, 0x00, sizeof(float)*N*dim);
    verlet_at2_cu<<<Tx_cu,Ty_cu>>>(dim, marr_d, xarr_d, xarr0_d, dxarr_d, dt, G, N, cut); // dx: acc
    gather_dx_cu<<<Tx_cu,Ty_cu>>>(dxarr_d, xarr_d, xarr0_d, N*dim);
    tmp = xarr_d;
    xarr_d = xarr0_d;
    xarr0_d = tmp;
    vec_add_cu<<<Tx_cu,Ty_cu>>>(xarr_d, xarr0_d, dxarr_d, N*dim);

    cudaDeviceSynchronize();
    // cudaMemcpy(xarr, xarr_d, sizeof(float)*N*dim, cudaMemcpyDeviceToHost);

    #ifdef GUI
    // copy x to host
    cudaMemcpy(xarr, xarr_d, sizeof(float)*N*dim, cudaMemcpyDeviceToHost);
    #endif
}

// cuda finalize program
void finalize_cu(){
    // free
    printf("cuda finalize\n");
    gpuErrchk( cudaFree(marr_d) );
    gpuErrchk( cudaFree(xarr_d) );
    gpuErrchk( cudaFree(xarr0_d) );
    gpuErrchk( cudaFree(dxarr_d) );
}

