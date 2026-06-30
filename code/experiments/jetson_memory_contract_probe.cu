#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

__global__ void store_kernel(int *ptr, int value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        ptr[0] = value;
}

static std::string json_escape(const std::string &input)
{
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out << "\\u";
                const char *hex = "0123456789abcdef";
                out << "00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

static std::string cuda_error_name(cudaError_t err)
{
    return err == cudaSuccess ? "cudaSuccess" : cudaGetErrorName(err);
}

static std::string cuda_error_text(cudaError_t err)
{
    return err == cudaSuccess ? "" : cudaGetErrorString(err);
}

static void print_probe_result(const char *name, cudaError_t err,
                               const std::string &extra = "")
{
    std::cout << "    \"" << name << "\": {"
              << "\"status\":\"" << (err == cudaSuccess ? "pass" : "fail")
              << "\",\"cuda_error\":\"" << json_escape(cuda_error_name(err))
              << "\",\"cuda_error_text\":\""
              << json_escape(cuda_error_text(err)) << "\"";
    if (!extra.empty())
        std::cout << "," << extra;
    std::cout << "}";
}

static void print_skipped_probe(const char *name, const std::string &reason)
{
    std::cout << "    \"" << name << "\": {"
              << "\"status\":\"skipped\","
              << "\"reason\":\"" << json_escape(reason) << "\"}";
}

static int get_attr(int device, cudaDeviceAttr attr)
{
    int value = 0;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, device);
    return err == cudaSuccess ? value : -1;
}

static cudaError_t run_store_kernel(int *device_ptr, int expected,
                                    int *host_ptr, bool *value_ok)
{
    if (value_ok)
        *value_ok = false;
    store_kernel<<<1, 1>>>(device_ptr, expected);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return err;
    if (value_ok)
        *value_ok = host_ptr && host_ptr[0] == expected;
    return cudaSuccess;
}

int main()
{
    const size_t bytes = 4096;
    int runtime_version = 0;
    int driver_version = 0;
    cudaError_t runtime_rc = cudaRuntimeGetVersion(&runtime_version);
    cudaError_t driver_rc = cudaDriverGetVersion(&driver_version);
    cudaError_t flags_rc = cudaSetDeviceFlags(cudaDeviceMapHost);

    int device_count = 0;
    cudaError_t count_rc = cudaGetDeviceCount(&device_count);
    int selected_device = device_count > 0 ? 0 : -1;
    if (selected_device >= 0)
        (void)cudaSetDevice(selected_device);

    std::cout << "{\n";
    std::cout << "  \"schema_version\": 1,\n";
    std::cout << "  \"cuda_runtime_version\": " << runtime_version << ",\n";
    std::cout << "  \"cuda_driver_version\": " << driver_version << ",\n";
    std::cout << "  \"cuda_runtime_version_status\": \""
              << (runtime_rc == cudaSuccess ? "pass" : "fail") << "\",\n";
    std::cout << "  \"cuda_driver_version_status\": \""
              << (driver_rc == cudaSuccess ? "pass" : "fail") << "\",\n";
    std::cout << "  \"cuda_set_device_flags\": {\"status\":\""
              << (flags_rc == cudaSuccess ? "pass" : "fail")
              << "\",\"cuda_error\":\"" << json_escape(cuda_error_name(flags_rc))
              << "\",\"cuda_error_text\":\""
              << json_escape(cuda_error_text(flags_rc)) << "\"},\n";
    std::cout << "  \"cuda_get_device_count\": {\"status\":\""
              << (count_rc == cudaSuccess ? "pass" : "fail")
              << "\",\"cuda_error\":\"" << json_escape(cuda_error_name(count_rc))
              << "\",\"cuda_error_text\":\""
              << json_escape(cuda_error_text(count_rc)) << "\"},\n";
    std::cout << "  \"device_count\": " << device_count << ",\n";
    std::cout << "  \"devices\": [";
    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        cudaError_t prop_rc = cudaGetDeviceProperties(&prop, dev);
        if (dev > 0)
            std::cout << ",";
        std::cout << "\n    {";
        std::cout << "\"device\":" << dev << ",";
        std::cout << "\"properties_status\":\""
                  << (prop_rc == cudaSuccess ? "pass" : "fail") << "\",";
        std::cout << "\"name\":\"" << json_escape(prop.name) << "\",";
        std::cout << "\"compute_capability\":\"" << prop.major << "."
                  << prop.minor << "\",";
        std::cout << "\"integrated\":" << prop.integrated << ",";
        std::cout << "\"unified_addressing\":" << prop.unifiedAddressing << ",";
        std::cout << "\"managed_memory\":" << prop.managedMemory << ",";
        std::cout << "\"host_register_supported\":"
                  << prop.hostRegisterSupported << ",";
        std::cout << "\"pageable_memory_access\":"
                  << prop.pageableMemoryAccess << ",";
        std::cout << "\"pageable_memory_access_uses_host_page_tables\":"
                  << prop.pageableMemoryAccessUsesHostPageTables << ",";
        std::cout << "\"concurrent_managed_access\":"
                  << prop.concurrentManagedAccess << ",";
        std::cout << "\"direct_managed_mem_access_from_host\":"
                  << prop.directManagedMemAccessFromHost << ",";
        std::cout << "\"can_use_host_pointer_for_registered_mem\":"
                  << get_attr(dev, cudaDevAttrCanUseHostPointerForRegisteredMem)
                  << ",";
        std::cout << "\"gpudirect_rdma_supported\":"
                  << get_attr(dev, cudaDevAttrGPUDirectRDMASupported) << ",";
        std::cout << "\"gpudirect_rdma_flush_writes_options\":"
                  << get_attr(dev, cudaDevAttrGPUDirectRDMAFlushWritesOptions)
                  << ",";
        std::cout << "\"gpudirect_rdma_writes_ordering\":"
                  << get_attr(dev, cudaDevAttrGPUDirectRDMAWritesOrdering);
        std::cout << "}";
    }
    std::cout << "\n  ],\n";
    std::cout << "  \"selected_device\": " << selected_device << ",\n";
    std::cout << "  \"probes\": {\n";

    bool first_probe = true;
    auto comma = [&first_probe]() {
        if (!first_probe)
            std::cout << ",\n";
        first_probe = false;
    };

    if (selected_device < 0) {
        comma(); print_skipped_probe("cudaHostAlloc", "no CUDA device");
        comma(); print_skipped_probe("cudaHostAllocMappedKernel", "no CUDA device");
        comma(); print_skipped_probe("cudaHostRegister", "no CUDA device");
        comma(); print_skipped_probe("cudaMallocManaged", "no CUDA device");
        comma(); print_skipped_probe("cudaMemPrefetchAsyncManaged", "no CUDA device");
        comma(); print_skipped_probe("cudaStreamAttachMemAsyncManaged", "no CUDA device");
        comma(); print_skipped_probe("pageableMemoryPrefetch", "no CUDA device");
    } else {
        int *host_alloc = nullptr;
        cudaError_t host_alloc_rc = cudaHostAlloc(
            reinterpret_cast<void **>(&host_alloc), bytes,
            cudaHostAllocDefault);
        comma(); print_probe_result("cudaHostAlloc", host_alloc_rc);
        if (host_alloc_rc == cudaSuccess)
            cudaFreeHost(host_alloc);

        int *mapped_host = nullptr;
        cudaError_t mapped_rc = cudaHostAlloc(
            reinterpret_cast<void **>(&mapped_host), bytes,
            cudaHostAllocMapped);
        int *mapped_device = nullptr;
        cudaError_t mapped_ptr_rc = cudaSuccess;
        cudaError_t mapped_kernel_rc = cudaSuccess;
        bool mapped_value_ok = false;
        if (mapped_rc == cudaSuccess) {
            mapped_host[0] = 0;
            mapped_ptr_rc = cudaHostGetDevicePointer(
                reinterpret_cast<void **>(&mapped_device), mapped_host, 0);
            if (mapped_ptr_rc == cudaSuccess)
                mapped_kernel_rc = run_store_kernel(mapped_device, 17,
                                                    mapped_host,
                                                    &mapped_value_ok);
        }
        std::ostringstream mapped_extra;
        mapped_extra << "\"host_alloc_mapped_error\":\""
                     << json_escape(cuda_error_name(mapped_rc)) << "\",";
        mapped_extra << "\"host_get_device_pointer_error\":\""
                     << json_escape(cuda_error_name(mapped_ptr_rc)) << "\",";
        mapped_extra << "\"value_observed\":"
                     << (mapped_value_ok ? "true" : "false");
        comma(); print_probe_result(
            "cudaHostAllocMappedKernel",
            mapped_rc == cudaSuccess && mapped_ptr_rc == cudaSuccess
                ? mapped_kernel_rc : (mapped_rc != cudaSuccess ? mapped_rc
                                      : mapped_ptr_rc),
            mapped_extra.str());
        if (mapped_rc == cudaSuccess)
            cudaFreeHost(mapped_host);

        void *registered_host = nullptr;
        int posix_rc = posix_memalign(&registered_host, 4096, bytes);
        cudaError_t register_rc = cudaErrorMemoryAllocation;
        cudaError_t register_ptr_rc = cudaErrorNotMapped;
        int *registered_device = nullptr;
        cudaError_t register_kernel_rc = cudaSuccess;
        bool register_value_ok = false;
        if (posix_rc == 0 && registered_host) {
            memset(registered_host, 0, bytes);
            register_rc = cudaHostRegister(registered_host, bytes,
                                           cudaHostRegisterDefault);
            if (register_rc == cudaSuccess) {
                register_ptr_rc = cudaHostGetDevicePointer(
                    reinterpret_cast<void **>(&registered_device),
                    registered_host, 0);
                if (register_ptr_rc == cudaSuccess)
                    register_kernel_rc = run_store_kernel(
                        registered_device, 23,
                        reinterpret_cast<int *>(registered_host),
                        &register_value_ok);
            }
        }
        std::ostringstream register_extra;
        register_extra << "\"posix_memalign_rc\":" << posix_rc << ",";
        register_extra << "\"host_get_device_pointer_error\":\""
                       << json_escape(cuda_error_name(register_ptr_rc)) << "\",";
        register_extra << "\"mapped_kernel_error\":\""
                       << json_escape(cuda_error_name(register_kernel_rc)) << "\",";
        register_extra << "\"value_observed\":"
                       << (register_value_ok ? "true" : "false");
        comma(); print_probe_result("cudaHostRegister", register_rc,
                                    register_extra.str());
        if (register_rc == cudaSuccess)
            cudaHostUnregister(registered_host);
        free(registered_host);

        int *managed = nullptr;
        cudaError_t managed_rc = cudaMallocManaged(
            reinterpret_cast<void **>(&managed), bytes);
        comma(); print_probe_result("cudaMallocManaged", managed_rc);
        cudaStream_t stream = nullptr;
        cudaError_t stream_rc = cudaStreamCreate(&stream);
        if (managed_rc == cudaSuccess) {
            managed[0] = 31;
            cudaMemLocation device_location;
            memset(&device_location, 0, sizeof(device_location));
            device_location.type = cudaMemLocationTypeDevice;
            device_location.id = selected_device;
            cudaError_t prefetch_rc = cudaMemPrefetchAsync(
                managed, bytes, device_location, 0, stream);
            if (prefetch_rc == cudaSuccess)
                prefetch_rc = cudaStreamSynchronize(stream);
            comma(); print_probe_result("cudaMemPrefetchAsyncManaged",
                                        prefetch_rc);
            cudaError_t attach_rc = cudaStreamAttachMemAsync(
                stream, managed, 0, cudaMemAttachGlobal);
            if (attach_rc == cudaSuccess)
                attach_rc = cudaStreamSynchronize(stream);
            comma(); print_probe_result("cudaStreamAttachMemAsyncManaged",
                                        attach_rc);
        } else {
            comma(); print_skipped_probe("cudaMemPrefetchAsyncManaged",
                                         "cudaMallocManaged failed");
            comma(); print_skipped_probe("cudaStreamAttachMemAsyncManaged",
                                         "cudaMallocManaged failed");
        }

        cudaDeviceProp selected_prop;
        memset(&selected_prop, 0, sizeof(selected_prop));
        cudaError_t selected_prop_rc =
            cudaGetDeviceProperties(&selected_prop, selected_device);
        if (selected_prop_rc == cudaSuccess &&
            selected_prop.pageableMemoryAccess) {
            void *pageable = malloc(bytes);
            cudaError_t pageable_rc = cudaErrorMemoryAllocation;
            if (pageable) {
                memset(pageable, 0, bytes);
                cudaMemLocation device_location;
                memset(&device_location, 0, sizeof(device_location));
                device_location.type = cudaMemLocationTypeDevice;
                device_location.id = selected_device;
                pageable_rc = cudaMemPrefetchAsync(
                    pageable, bytes, device_location, 0, stream);
                if (pageable_rc == cudaSuccess)
                    pageable_rc = cudaStreamSynchronize(stream);
            }
            comma(); print_probe_result("pageableMemoryPrefetch", pageable_rc);
            free(pageable);
        } else {
            comma(); print_skipped_probe(
                "pageableMemoryPrefetch",
                "selected device does not report pageableMemoryAccess");
        }

        if (managed_rc == cudaSuccess)
            cudaFree(managed);
        if (stream_rc == cudaSuccess)
            cudaStreamDestroy(stream);
    }

    std::cout << "\n  }\n";
    std::cout << "}\n";
    return 0;
}
