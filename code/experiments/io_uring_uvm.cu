#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <stdint.h>
#include <string.h>

#define BUF_SIZE (4096 * 1024)
#define QD 16
#define CHECKSUM_THREADS 256

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ static void checksum_kernel(const unsigned char *buf,
                                       size_t size,
                                       unsigned long long *partials) {
    __shared__ unsigned long long local[CHECKSUM_THREADS];
    const size_t tid = threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = 0;

    while (idx < size) {
        acc += (unsigned long long)buf[idx] * (unsigned long long)((idx % 251U) + 1U);
        idx += stride;
    }

    local[tid] = acc;
    __syncthreads();

    for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            local[tid] += local[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = local[0];
    }
}

static unsigned long long cpu_checksum(const unsigned char *buf, size_t size) {
    unsigned long long acc = 0;
    for (size_t i = 0; i < size; ++i) {
        acc += (unsigned long long)buf[i] * (unsigned long long)((i % 251U) + 1U);
    }
    return acc;
}

static unsigned long long fold_partials(const unsigned long long *partials,
                                        int blocks) {
    unsigned long long acc = 0;
    for (int i = 0; i < blocks; ++i) {
        acc += partials[i];
    }
    return acc;
}

static const char *memory_type_name(cudaMemoryType type) {
    switch (type) {
        case cudaMemoryTypeUnregistered:
            return "unregistered";
        case cudaMemoryTypeHost:
            return "host";
        case cudaMemoryTypeDevice:
            return "device";
        case cudaMemoryTypeManaged:
            return "managed";
        default:
            return "unknown";
    }
}

static const char *location_type_name(int value) {
    switch (value) {
        case cudaMemLocationTypeInvalid:
            return "invalid";
        case cudaMemLocationTypeDevice:
            return "device";
        case cudaMemLocationTypeHost:
            return "host";
        default:
            return "unknown";
    }
}

static void dump_pointer_attributes(const char *label, const void *ptr) {
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        printf("[!] POINTER_ATTR %s error=%s\n", label, cudaGetErrorString(err));
        return;
    }

    printf("[+] POINTER_ATTR %s type=%s device=%d host_ptr=%p device_ptr=%p\n",
           label,
           memory_type_name(attrs.type),
           attrs.device,
           attrs.hostPointer,
           attrs.devicePointer);
}

static void dump_host_flags(const char *label, const void *ptr) {
    unsigned int flags = 0;
    cudaError_t err = cudaHostGetFlags(&flags, (void *)ptr);
    if (err != cudaSuccess) {
        printf("[!] HOST_FLAGS %s error=%s\n", label, cudaGetErrorString(err));
        return;
    }

    printf("[+] HOST_FLAGS %s raw=0x%x portable=%d mapped=%d writecombined=%d\n",
           label,
           flags,
           (flags & cudaHostAllocPortable) != 0,
           (flags & cudaHostAllocMapped) != 0,
           (flags & cudaHostAllocWriteCombined) != 0);
}

static void dump_range_attribute_u32(const char *label,
                                     const void *ptr,
                                     size_t size,
                                     cudaMemRangeAttribute attr,
                                     const char *attr_name) {
    unsigned int value = 0;
    cudaError_t err = cudaMemRangeGetAttribute(&value, sizeof(value), attr, ptr, size);
    if (err != cudaSuccess) {
        printf("[!] RANGE_ATTR %s %s error=%s\n",
               label,
               attr_name,
               cudaGetErrorString(err));
        (void)cudaGetLastError();
        return;
    }

    if (attr == cudaMemRangeAttributePreferredLocationType ||
        attr == cudaMemRangeAttributeLastPrefetchLocationType) {
        printf("[+] RANGE_ATTR %s %s=%u(%s)\n",
               label,
               attr_name,
               value,
               location_type_name((int)value));
        return;
    }

    printf("[+] RANGE_ATTR %s %s=%u\n", label, attr_name, value);
}

static void dump_memory_diagnostics(const char *label, const void *ptr, size_t size) {
    dump_pointer_attributes(label, ptr);
    dump_host_flags(label, ptr);
    dump_range_attribute_u32(label, ptr, size,
                             cudaMemRangeAttributePreferredLocationType,
                             "preferred_location_type");
    dump_range_attribute_u32(label, ptr, size,
                             cudaMemRangeAttributePreferredLocationId,
                             "preferred_location_id");
    dump_range_attribute_u32(label, ptr, size,
                             cudaMemRangeAttributeLastPrefetchLocationType,
                             "last_prefetch_location_type");
    dump_range_attribute_u32(label, ptr, size,
                             cudaMemRangeAttributeLastPrefetchLocationId,
                             "last_prefetch_location_id");
}

int main(int argc, char *argv[]) {
    bool use_managed_buffer = false;
    const char *path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--managed-buffer") == 0) {
            use_managed_buffer = true;
        } else if (!path) {
            path = argv[i];
        } else {
            fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (!path) {
        printf("Usage: %s [--managed-buffer] <file>\n", argv[0]);
        return 1;
    }

    int rc = 1;
    void *buf = NULL;
    struct io_uring ring;
    bool ring_ready = false;
    bool host_registered = false;
    bool managed_allocated = false;
    void *device_ptr = NULL;
    struct io_uring_sqe *sqe = NULL;
    struct io_uring_cqe *cqe = NULL;
    size_t read_bytes = 0;
    const int blocks = 256;
    unsigned long long *device_partials = NULL;
    unsigned long long *host_partials = NULL;
    unsigned long long cpu = 0;
    unsigned long long gpu = 0;

    int open_flags = O_RDONLY | (use_managed_buffer ? 0 : O_DIRECT);
    int fd = open(path, open_flags);
    if (fd < 0) {
        perror(use_managed_buffer ? "open buffered" : "open O_DIRECT");
        return 1;
    }
    if (use_managed_buffer) {
        printf("[*] Allocating managed buffer...\n");
        check_cuda(cudaMallocManaged(&buf, BUF_SIZE), "cudaMallocManaged");
        managed_allocated = true;
        device_ptr = buf;
        printf("[+] Managed buffer allocated at: %p\n", buf);
        dump_memory_diagnostics("managed_ptr_before_read", buf, BUF_SIZE);

        printf("[*] Issuing buffered pread into managed buffer...\n");
        ssize_t rd = pread(fd, buf, BUF_SIZE, 0);
        if (rd < 0) {
            perror("pread managed");
            goto out_close;
        }
        if (rd == 0) {
            fprintf(stderr, "pread managed returned zero bytes\n");
            goto out_close;
        }
        read_bytes = (size_t)rd;
        printf("[+] Buffered pread completed: %zd bytes\n", rd);
        dump_memory_diagnostics("managed_ptr_after_read", buf, read_bytes);

        int dev = 0;
        check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
        cudaMemLocation dev_loc = { cudaMemLocationTypeDevice, dev };
        printf("[*] Prefetching managed storage buffer to device %d...\n", dev);
        check_cuda(cudaMemPrefetchAsync(buf, read_bytes, dev_loc, 0), "cudaMemPrefetchAsync device");
        check_cuda(cudaDeviceSynchronize(), "managed prefetch sync");
        dump_memory_diagnostics("managed_ptr_after_prefetch_device", buf, read_bytes);

        printf("[*] Prefetching managed storage buffer back to host...\n");
        cudaMemLocation host_loc = { cudaMemLocationTypeHost, 0 };
        check_cuda(cudaMemPrefetchAsync(buf, read_bytes, host_loc, 0), "cudaMemPrefetchAsync host");
        check_cuda(cudaDeviceSynchronize(), "managed prefetch host sync");
        dump_memory_diagnostics("managed_ptr_after_prefetch_host", buf, read_bytes);
    } else {
        if (io_uring_queue_init(QD, &ring, IORING_SETUP_IOPOLL) < 0) {
            perror("io_uring_queue_init");
            goto out_close;
        }
        ring_ready = true;

        if (posix_memalign(&buf, 4096, BUF_SIZE) != 0) {
            perror("posix_memalign");
            goto out_close;
        }

        printf("[*] Pinning memory via cudaHostRegister...\n");
        check_cuda(cudaHostRegister(buf, BUF_SIZE, cudaHostRegisterDefault), "cudaHostRegister");
        host_registered = true;

        check_cuda(cudaHostGetDevicePointer(&device_ptr, buf, 0), "cudaHostGetDevicePointer");
        printf("[+] Memory pinned and mapped to device pointer: %p\n", device_ptr);
        dump_memory_diagnostics("host_ptr", buf, BUF_SIZE);
        dump_pointer_attributes("device_alias", device_ptr);

        printf("[*] Submitting io_uring read request...\n");
        sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buf, BUF_SIZE, 0);
        io_uring_submit(&ring);

        printf("[*] Polling for completion...\n");
        if (io_uring_wait_cqe(&ring, &cqe) < 0) {
            perror("io_uring_wait_cqe");
            goto out_close;
        }

        if (cqe->res < 0) {
            fprintf(stderr, "Async read failed: %d\n", cqe->res);
            io_uring_cqe_seen(&ring, cqe);
            goto out_close;
        } else if (cqe->res == 0) {
            fprintf(stderr, "Async read returned zero bytes\n");
            io_uring_cqe_seen(&ring, cqe);
            goto out_close;
        } else {
            printf("[+] Async read completed: %d bytes\n", cqe->res);
        }

        read_bytes = (size_t)cqe->res;
        io_uring_cqe_seen(&ring, cqe);
    }

    host_partials =
        (unsigned long long *)calloc((size_t)blocks, sizeof(unsigned long long));
    if (!host_partials) {
        perror("calloc");
        goto out_close;
    }
    check_cuda(cudaMalloc((void **)&device_partials,
                          (size_t)blocks * sizeof(unsigned long long)),
               "cudaMalloc partials");
    check_cuda(cudaMemset(device_partials, 0,
                          (size_t)blocks * sizeof(unsigned long long)),
               "cudaMemset partials");

    printf("[*] Launching GPU checksum over the same storage-filled %s buffer...\n",
           use_managed_buffer ? "managed" : "mapped");
    checksum_kernel<<<blocks, CHECKSUM_THREADS>>>(
        (const unsigned char *)device_ptr, read_bytes, device_partials);
    check_cuda(cudaGetLastError(), "checksum_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "checksum_kernel sync");
    check_cuda(cudaMemcpy(host_partials, device_partials,
                          (size_t)blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy partials");

    cpu = cpu_checksum((const unsigned char *)buf, read_bytes);
    gpu = fold_partials(host_partials, blocks);

    printf("[+] CPU checksum over storage buffer: 0x%016llx\n", cpu);
    printf("[+] GPU %s-buffer checksum:      0x%016llx\n",
           use_managed_buffer ? "managed" : "mapped",
           gpu);

    if (gpu != cpu) {
        fprintf(stderr, "CHECKSUM_MISMATCH: GPU did not observe the storage-filled buffer\n");
        goto out_close;
    }
    if (use_managed_buffer) {
        printf("[+] CHECKSUM_MATCH: GPU observed the same storage-filled managed buffer\n");
    } else {
        printf("[+] CHECKSUM_MATCH: GPU observed the same O_DIRECT storage-filled pinned buffer\n");
    }

    rc = 0;
    
out_close:
    if (device_partials) {
        cudaFree(device_partials);
    }
    free(host_partials);
    if (host_registered) {
        cudaHostUnregister(buf);
    }
    if (managed_allocated) {
        cudaFree(buf);
    } else {
        free(buf);
    }
    if (ring_ready) {
        io_uring_queue_exit(&ring);
    }
    close(fd);
    return rc;
}
