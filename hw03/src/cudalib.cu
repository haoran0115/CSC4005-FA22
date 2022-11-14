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

__global__ void verlet_add_cu(float *a, float *b, float *c, int N, int dim,
    int xmin, int xmax, int ymin, int ymax){
    int size = blockDim.x * gridDim.x;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int start_idx, end_idx;
    partition_d(N, size, idx, &start_idx, &end_idx);
    for (int i = start_idx; i < end_idx; i++){
        float x = b[i*dim+0] + c[i*dim+0];
        float y = b[i*dim+1] + c[i*dim+1];
        if (x < xmin) x += 2 * (xmin - x);
        else if (x > xmax) x += 2 * (xmax - x);
        if (y < ymin) y += 2 * (ymin - y);
        else if (y > ymax) y += 2 * (ymax - y);
        a[i*dim+0] = x;
        a[i*dim+1] = y;
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

__global__ void verlet_at2_cu(const int dim, float *marr, float *xarr, float *xarr0,
    float *dxarr, float dt, float G, int N, float cut){
    // partition
    int size = gridDim.x;
    int idx = blockIdx.x;
    int block_start_idx, block_end_idx;
    partition_d(N, size, idx, &block_start_idx, &block_end_idx);
    // if (threadIdx.x==0) printf("%d %d\n", block_start_idx, block_end_idx);
    // shared memory
    __shared__ float     marr_t[BLOCK_SIZE];
    __shared__ float xarr_l_t[BLOCK_SIZE*2];
    __shared__ float xarr_g_t[BLOCK_SIZE*2];
    __shared__ float  dxarr_t[BLOCK_SIZE*2];
    for (int i = block_start_idx; i < block_end_idx; i+=BLOCK_SIZE){
        // tmp variables
        float tmpx = 0.0;
        float tmpy = 0.0;
        if (i + threadIdx.x < block_end_idx){
            // get local coords
            xarr_l_t[threadIdx.x*dim+0] = xarr[i*dim+threadIdx.x*dim+0];
            xarr_l_t[threadIdx.x*dim+1] = xarr[i*dim+threadIdx.x*dim+1];
        }
        __syncthreads();
        // N loop
        for (int j = 0; j < N; j+=BLOCK_SIZE){
            if (threadIdx.x + j < N){
                marr_t[threadIdx.x] = marr[threadIdx.x+j];
                xarr_g_t[threadIdx.x*dim+0] = xarr[threadIdx.x*dim+j*dim+0];
                xarr_g_t[threadIdx.x*dim+1] = xarr[threadIdx.x*dim+j*dim+1];
            }
            __syncthreads();
            // if (blockIdx.x==0 && threadIdx.x==0 && j==0) print_arr_d(xarr_g_t, 8);
            for (int k = 0; k < BLOCK_SIZE; k++){
            if (k + j < N && threadIdx.x + j < N){
                // compute xij
                float xij0 = xarr_g_t[k*dim+0] - xarr_l_t[threadIdx.x*dim+0];
                float xij1 = xarr_g_t[k*dim+1] - xarr_l_t[threadIdx.x*dim+1];
                float rij = sqrt(xij0*xij0 + xij1*xij1);
                if (rij < cut) rij = cut;
                tmpx += xij0/(rij*rij*rij) * marr_t[k]* G*dt*dt;
                tmpy += xij1/(rij*rij*rij) * marr_t[k]* G*dt*dt;
            }}
            // assign value to shared memory
        }
        if (i + threadIdx.x < block_end_idx){
            // assign value back to global memory 
            dxarr_t[threadIdx.x*dim+0] = tmpx;
            dxarr_t[threadIdx.x*dim+1] = tmpy;
        }
        __syncthreads();
        if (i + threadIdx.x < block_end_idx){
            dxarr[threadIdx.x*dim+i*dim+0] = dxarr_t[threadIdx.x*dim+0];
            dxarr[threadIdx.x*dim+i*dim+1] = dxarr_t[threadIdx.x*dim+1];
        }
        __syncthreads();
    }
}

// cuda initialize program
void initialize_cu(float *marr, float *xarr, int N, int dim, int Tx, int Ty,
    float xmin, float xmax, float ymin, float ymax){
    printf("cuda initialize\n");
    // cuda parameters
    Tx_cu = Tx;
    Ty_cu = Ty;
    xmin_d = xmin;
    xmax_d = xmax;
    ymin_d = ymin;
    ymax_d = ymax;
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
    cudaDeviceSynchronize();
    verlet_at2_cu<<<32,BLOCK_SIZE>>>(dim, marr_d, xarr_d, xarr0_d, dxarr_d, dt, G, N, cut); // dx: acc
    cudaDeviceSynchronize();
    gather_dx_cu<<<Tx_cu,Ty_cu>>>(dxarr_d, xarr_d, xarr0_d, N*dim);
    cudaDeviceSynchronize();
    tmp = xarr_d;
    xarr_d = xarr0_d;
    xarr0_d = tmp;
    verlet_add_cu<<<Tx_cu,Ty_cu>>>(xarr_d, xarr0_d, dxarr_d, N, dim, xmin_d, xmax_d, ymin_d, ymax_d);

    cudaDeviceSynchronize();
    cudaMemcpy(xarr, xarr_d, sizeof(float)*N*dim, cudaMemcpyDeviceToHost);

    #ifdef GUI
    // copy x to host
    cudaMemcpy(xarr, xarr_d, sizeof(float)*N*dim, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
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

