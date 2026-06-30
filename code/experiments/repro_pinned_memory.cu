#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}

int main() {
    int *h_data;
    cudaError_t err = cudaMallocHost((void**)&h_data, 1024 * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int *d_data;
    err = cudaHostGetDevicePointer((void**)&d_data, h_data, 0);
    if (err != cudaSuccess) {
        printf("cudaHostGetDevicePointer failed: %s (Error code: %d)\n", cudaGetErrorString(err), err);
        printf("On Tegra UMA architectures, this returns cudaErrorNotSupported (71).\n");
    } else {
        printf("Pointer acquired successfully. Attempting kernel launch...\n");
        testKernel<<<1, 1024>>>(d_data);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
             printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFreeHost(h_data);
    return 0;
}
