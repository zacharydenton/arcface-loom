// Test-only HIP interposer, loaded solely by test_native_errors.py in a child.
// Real copies and synchronization still run. Even a simulated drain failure
// finishes the actual transfer so the test never strands DMA to host memory.
#include <hip/hip_runtime.h>
#include <dlfcn.h>

namespace {
bool fail_launch = false, fail_drain = false;
int synchronizations = 0, uploads = 0, launches = 0;
}

extern "C" void arcface_test_arm(int drain_error) {
    fail_launch = true;
    fail_drain = drain_error != 0;
    synchronizations = uploads = launches = 0;
}

extern "C" int arcface_test_count(int kind) {
    return kind == 0 ? synchronizations : kind == 1 ? uploads : launches;
}

extern "C" hipError_t hipMemcpyAsync(void *dst, const void *src, size_t bytes,
                                     hipMemcpyKind kind, hipStream_t stream) {
    static auto real = reinterpret_cast<decltype(&hipMemcpyAsync)>(dlsym(RTLD_NEXT, "hipMemcpyAsync"));
    if (kind == hipMemcpyHostToDevice) ++uploads;
    return real(dst, src, bytes, kind, stream);
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, hipStream_t stream, void **params, void **extra) {
    ++launches;
    if (fail_launch) {
        fail_launch = false;
        return hipErrorInvalidValue;
    }
    static auto real = reinterpret_cast<decltype(&hipModuleLaunchKernel)>(dlsym(RTLD_NEXT, "hipModuleLaunchKernel"));
    return real(function, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
    ++synchronizations;
    static auto real = reinterpret_cast<decltype(&hipStreamSynchronize)>(dlsym(RTLD_NEXT, "hipStreamSynchronize"));
    hipError_t result = real(stream);
    if (fail_drain) {
        fail_drain = false;
        return hipErrorUnknown;
    }
    return result;
}
