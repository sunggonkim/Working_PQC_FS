#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

__global__ static void qos_contract_kernel(unsigned long long *out,
                                           unsigned int iterations)
{
    unsigned long long value = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < iterations; ++i)
        value = value * 1664525ULL + 1013904223ULL + (unsigned long long)i;
    out[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

static const char *status(cudaError_t rc)
{
    return rc == cudaSuccess ? "pass" : "fail";
}

static void print_result(cudaError_t rc)
{
    std::printf("{\"status\":\"%s\",\"code\":%d,\"error\":\"%s\"}",
                status(rc), (int)rc, cudaGetErrorString(rc));
}

int main(void)
{
    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);
    std::printf("{\"schema_version\":1,");
    std::printf("\"device_count\":{\"status\":\"%s\",\"code\":%d,\"value\":%d,\"error\":\"%s\"}",
                status(rc), (int)rc, device_count, cudaGetErrorString(rc));
    if (rc != cudaSuccess || device_count <= 0) {
        std::printf(",\"overall_pass\":false}\n");
        return 0;
    }

    int device = 0;
    rc = cudaSetDevice(device);
    std::printf(",\"set_device\":{\"status\":\"%s\",\"code\":%d,\"device\":%d,\"error\":\"%s\"}",
                status(rc), (int)rc, device, cudaGetErrorString(rc));
    if (rc != cudaSuccess) {
        std::printf(",\"overall_pass\":false}\n");
        return 0;
    }

    cudaDeviceProp prop;
    rc = cudaGetDeviceProperties(&prop, device);
    std::printf(",\"device\":{\"status\":\"%s\",\"name\":\"%s\",\"major\":%d,\"minor\":%d,"
                "\"multiProcessorCount\":%d,\"concurrentKernels\":%d,"
                "\"managedMemory\":%d,\"unifiedAddressing\":%d}",
                status(rc), rc == cudaSuccess ? prop.name : "", rc == cudaSuccess ? prop.major : 0,
                rc == cudaSuccess ? prop.minor : 0, rc == cudaSuccess ? prop.multiProcessorCount : 0,
                rc == cudaSuccess ? prop.concurrentKernels : 0,
                rc == cudaSuccess ? prop.managedMemory : 0,
                rc == cudaSuccess ? prop.unifiedAddressing : 0);

    int least_priority = 0;
    int greatest_priority = 0;
    cudaError_t priority_rc = cudaDeviceGetStreamPriorityRange(
        &least_priority, &greatest_priority);
    std::printf(",\"stream_priority_range\":{\"status\":\"%s\",\"code\":%d,"
                "\"least_priority\":%d,\"greatest_priority\":%d,"
                "\"distinct_range\":%s,\"error\":\"%s\"}",
                status(priority_rc), (int)priority_rc, least_priority, greatest_priority,
                least_priority != greatest_priority ? "true" : "false",
                cudaGetErrorString(priority_rc));

    cudaStream_t greatest_stream = nullptr;
    cudaStream_t least_stream = nullptr;
    cudaError_t greatest_stream_rc = cudaStreamCreateWithPriority(
        &greatest_stream, cudaStreamNonBlocking, greatest_priority);
    cudaError_t least_stream_rc = cudaStreamCreateWithPriority(
        &least_stream, cudaStreamNonBlocking, least_priority);
    std::printf(",\"stream_create_greatest_priority\":");
    print_result(greatest_stream_rc);
    std::printf(",\"stream_create_least_priority\":");
    print_result(least_stream_rc);

    const unsigned int blocks = 2;
    const unsigned int threads = 128;
    const unsigned int iterations = 32;
    const size_t elements = (size_t)blocks * (size_t)threads;
    const size_t bytes = elements * sizeof(unsigned long long);
    unsigned long long *managed = nullptr;
    cudaError_t malloc_rc = cudaMallocManaged(&managed, bytes);
    std::printf(",\"managed_alloc\":{\"status\":\"%s\",\"code\":%d,\"bytes\":%zu,\"error\":\"%s\"}",
                status(malloc_rc), (int)malloc_rc, bytes, cudaGetErrorString(malloc_rc));

    cudaError_t prefetch_device_rc = cudaErrorUnknown;
    cudaError_t attach_rc = cudaErrorUnknown;
    cudaError_t launch_error_rc = cudaErrorUnknown;
    cudaError_t event_elapsed_rc = cudaErrorUnknown;
    cudaError_t prefetch_host_rc = cudaErrorUnknown;
    cudaError_t stream_sync_rc = cudaErrorUnknown;
    cudaError_t device_sync_rc = cudaErrorUnknown;
    float kernel_elapsed_ms = -1.0f;
    bool verification_pass = false;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    cudaError_t event_create_start_rc = cudaEventCreate(&start_event);
    cudaError_t event_create_stop_rc = cudaEventCreate(&stop_event);
    cudaMemLocation device_location = { cudaMemLocationTypeDevice, device };
    cudaMemLocation host_location = { cudaMemLocationTypeHost, 0 };

    if (malloc_rc == cudaSuccess && greatest_stream_rc == cudaSuccess) {
        for (size_t i = 0; i < elements; ++i)
            managed[i] = 0ULL;

        prefetch_device_rc = cudaMemPrefetchAsync(
            managed, bytes, device_location, 0, greatest_stream);
        if (prefetch_device_rc == cudaSuccess) {
            attach_rc = cudaStreamAttachMemAsync(
                greatest_stream, managed, 0, cudaMemAttachGlobal);
        }
        if (attach_rc == cudaSuccess && event_create_start_rc == cudaSuccess &&
            event_create_stop_rc == cudaSuccess) {
            (void)cudaEventRecord(start_event, greatest_stream);
            qos_contract_kernel<<<blocks, threads, 0, greatest_stream>>>(
                managed, iterations);
            launch_error_rc = cudaGetLastError();
            (void)cudaEventRecord(stop_event, greatest_stream);
            prefetch_host_rc = cudaMemPrefetchAsync(
                managed, bytes, host_location, 0, greatest_stream);
            stream_sync_rc = cudaStreamSynchronize(greatest_stream);
            event_elapsed_rc = cudaEventElapsedTime(
                &kernel_elapsed_ms, start_event, stop_event);
            verification_pass = stream_sync_rc == cudaSuccess && managed[0] != 0ULL;
        }
    }
    device_sync_rc = cudaDeviceSynchronize();

    std::printf(",\"prefetch_device\":");
    print_result(prefetch_device_rc);
    std::printf(",\"attach_managed\":");
    print_result(attach_rc);
    std::printf(",\"kernel_launch_shape\":{\"kernel\":\"qos_contract_kernel\","
                "\"blocks\":%u,\"threads_per_block\":%u,\"dynamic_shared_bytes\":0,"
                "\"stream\":\"greatest_priority_stream\",\"iterations\":%u,"
                "\"launch_status\":\"%s\",\"launch_code\":%d,\"launch_error\":\"%s\"}",
                blocks, threads, iterations, status(launch_error_rc),
                (int)launch_error_rc, cudaGetErrorString(launch_error_rc));
    std::printf(",\"prefetch_host\":");
    print_result(prefetch_host_rc);
    std::printf(",\"synchronization\":{\"stream_synchronize\":{\"status\":\"%s\","
                "\"code\":%d,\"error\":\"%s\"},\"device_synchronize\":{\"status\":\"%s\","
                "\"code\":%d,\"error\":\"%s\"},\"event_elapsed\":{\"status\":\"%s\","
                "\"code\":%d,\"elapsed_ms\":%.6f,\"error\":\"%s\"}}",
                status(stream_sync_rc), (int)stream_sync_rc, cudaGetErrorString(stream_sync_rc),
                status(device_sync_rc), (int)device_sync_rc, cudaGetErrorString(device_sync_rc),
                status(event_elapsed_rc), (int)event_elapsed_rc, kernel_elapsed_ms,
                cudaGetErrorString(event_elapsed_rc));
    std::printf(",\"verification\":{\"status\":\"%s\",\"managed_first_value_nonzero\":%s}",
                verification_pass ? "pass" : "fail",
                verification_pass ? "true" : "false");

    if (start_event) (void)cudaEventDestroy(start_event);
    if (stop_event) (void)cudaEventDestroy(stop_event);
    if (managed) (void)cudaFree(managed);
    if (greatest_stream) (void)cudaStreamDestroy(greatest_stream);
    if (least_stream) (void)cudaStreamDestroy(least_stream);

    const bool overall_pass =
        priority_rc == cudaSuccess &&
        greatest_stream_rc == cudaSuccess &&
        least_stream_rc == cudaSuccess &&
        malloc_rc == cudaSuccess &&
        prefetch_device_rc == cudaSuccess &&
        attach_rc == cudaSuccess &&
        launch_error_rc == cudaSuccess &&
        prefetch_host_rc == cudaSuccess &&
        stream_sync_rc == cudaSuccess &&
        device_sync_rc == cudaSuccess &&
        event_elapsed_rc == cudaSuccess &&
        verification_pass;
    std::printf(",\"overall_pass\":%s}\n", overall_pass ? "true" : "false");
    return overall_pass ? 0 : 1;
}
