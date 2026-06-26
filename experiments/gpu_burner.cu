#include <cuda_runtime.h>
#include <stdio.h>

__global__ void burn_gpu(float *d_out, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)idx;
    for (int i = 0; i < iters; i++) {
        val = sinf(val) * cosf(val);
    }
    d_out[idx] = val;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <duration_sec>\n", argv[0]);
        return 1;
    }
    int duration = atoi(argv[1]);
    
    int blocks = 1024;
    int threads = 1024;
    float *d_out;
    cudaMalloc(&d_out, blocks * threads * sizeof(float));
    
    // Warmup
    burn_gpu<<<blocks, threads>>>(d_out, 100);
    cudaDeviceSynchronize();
    
    time_t start = time(NULL);
    while (time(NULL) - start < duration) {
        burn_gpu<<<blocks, threads>>>(d_out, 100000);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_out);
    return 0;
}
