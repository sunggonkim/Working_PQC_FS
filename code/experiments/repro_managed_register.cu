#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("[*] Starting repro_managed_register...\n");
    void *managed_ptr = NULL;
    size_t size = 16 * 1024 * 1024; // 16MB

    cudaError_t err = cudaMallocManaged(&managed_ptr, size);
    if (err != cudaSuccess) {
        printf("cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[+] cudaMallocManaged allocated 16MB at %p\n", managed_ptr);

    // Give bpftrace a moment to hook if needed, or we just run it straight.
    printf("[*] Attempting cudaHostRegister on managed memory...\n");
    err = cudaHostRegister(managed_ptr, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        printf("cudaHostRegister failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("[+] cudaHostRegister SUCCESS! The memory is pinned.\n");
        // Access it so page fault happens if it wasn't pinned
        int *data = (int*)managed_ptr;
        data[0] = 42;
    }

    // Keep it alive briefly for tracing
    sleep(2);

    cudaFree(managed_ptr);
    printf("[*] Done.\n");
    return 0;
}
