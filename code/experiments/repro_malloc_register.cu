#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("[*] Starting repro_malloc_register...\n");
    void *cpu_ptr = NULL;
    size_t size = 16 * 1024 * 1024; // 16MB

    if (posix_memalign(&cpu_ptr, 4096, size) != 0) {
        printf("posix_memalign failed\n");
        return 1;
    }
    printf("[+] posix_memalign allocated 16MB at %p\n", cpu_ptr);

    printf("[*] Attempting cudaHostRegister on CPU memory...\n");
    cudaError_t err = cudaHostRegister(cpu_ptr, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        printf("cudaHostRegister failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("[+] cudaHostRegister SUCCESS! The memory is pinned.\n");
        void *d_ptr = NULL;
        err = cudaHostGetDevicePointer(&d_ptr, cpu_ptr, 0);
        if (err != cudaSuccess) {
            printf("cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        } else {
            printf("[+] cudaHostGetDevicePointer SUCCESS! Device ptr: %p\n", d_ptr);
        }
    }

    free(cpu_ptr);
    printf("[*] Done.\n");
    return 0;
}
