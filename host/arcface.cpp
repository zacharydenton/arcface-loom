// w600k_r50 (ArcFace iResNet-50) forward pass on gfx1151: a resident session that
// walks the launch table tools/gen_launch_table.py generated from the ONNX graph.
//
// Nothing about the network is written here by hand. Each entry of the table
// names a kernel, its configuration, the buffers it reads and writes and the
// extra operands its epilogue takes; this file only knows how to build the
// kernarg block for each kernel kind.
//
// Build: ./scripts/build_host.sh  (host/arcface CLI and build/libarcface.so)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "arcface.h"

namespace {

constexpr int SIZE = 112;
constexpr int CHANNELS = 3;                 // BGR bytes per input pixel
constexpr int MAX_BATCH = 64;
constexpr int THREADS = 256;
constexpr int TILE = 64;                    // the M tile of every WMMA kernel

struct arcface_launch {
    int kind, variant, kernel;
    const char *name, *stage;
    int h, w, stride, cin_pad, cin_stride, k_size, n_size, ho, wo;
    int tile;                       // N tile of a 3x3 conv: 64, or 128 (tile_for in tools/export_weights.py)
    int splits;                     // head: split-K count
    int bias_rows;                  // 1, or 9 for the border-class table of a bnprelu conv
    int aux;                        // extra kernargs after (m, a, w, bias, c): bit 0 residual, bit 1 slope
    int src_buf, dst_buf, extra_buf;
};

#include "graph_table.inc"

enum { KIND_CONVERT = 0, KIND_CONV = 1, KIND_HEAD_MATMUL = 2, KIND_HEAD_REDUCE = 3 };
enum { AUX_RESIDUAL = 1, AUX_SLOPE = 2 };

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

struct Span { size_t offset, count; };

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string name; size_t offset, count;
    while (in >> name >> offset >> count) spans[name] = {offset, count};
    return spans;
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<char> buffer(in.tellg());
    in.seekg(0);
    in.read(buffer.data(), buffer.size());
    return buffer;
}

// Every span inside the blob, both multiplications overflow-checked.
void check_spans(const std::map<std::string, Span> &spans, size_t blob_bytes,
                 size_t element_bytes, const char *what) {
    for (const auto &e : spans) {
        size_t begin, extent, end;
        if (__builtin_mul_overflow(e.second.offset, element_bytes, &begin) ||
            __builtin_mul_overflow(e.second.count, element_bytes, &extent) ||
            __builtin_add_overflow(begin, extent, &end) || end > blob_bytes)
            throw std::runtime_error(std::string(what) + ": span '" + e.first + "' runs past the blob");
    }
}

// Every tensor a launch reads must exist with exactly the element count its
// kernel is compiled for: the kernels take bare pointers and never see counts.
size_t require(const std::map<std::string, Span> &spans, const std::string &name, size_t count,
               const char *what) {
    auto it = spans.find(name);
    if (it == spans.end()) throw std::runtime_error(std::string(what) + ": missing tensor '" + name + "'");
    if (it->second.count != count)
        throw std::runtime_error(std::string(what) + ": tensor '" + name + "' has " +
                                 std::to_string(it->second.count) + " elements, the kernel expects " +
                                 std::to_string(count));
    return it->second.offset;
}

struct Kernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    void load(const std::string &path, const char *symbol) {
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

// Loom's AMDGPU kernarg ABI: i32 scalars 4-byte aligned, pointers 8-byte aligned.
struct KernArgs {
    // Zero-initialized: `index` kernargs occupy 8 bytes here, and the split-K
    // matmul guards its A load on the raw m_size, so a garbage upper half would
    // read far out of bounds. scalar_i32 writes only the low 4 bytes.
    alignas(16) unsigned char bytes[128] = {};
    size_t size = 0;
    void scalar_i32(int v) { size = (size + 3) & ~size_t(3); memcpy(bytes + size, &v, 4); size += 4; }
    void pointer(const void *p) { size = (size + 7) & ~size_t(7); memcpy(bytes + size, &p, 8); size += 8; }
};

struct Profiler {
    bool enabled = false;
    hipStream_t stream = nullptr;
    std::map<std::string, double> stage_us;
    std::map<std::string, int> stage_calls;
};

struct Stage {
    Profiler &p; std::string name; std::chrono::steady_clock::time_point start;
    Stage(Profiler &prof, const char *n) : p(prof), name(n) {
        if (p.enabled) { HIP_CHECK(hipStreamSynchronize(p.stream)); start = std::chrono::steady_clock::now(); }
    }
    ~Stage() {
        if (!p.enabled) return;
        (void)hipStreamSynchronize(p.stream);   // a destructor must not throw
        auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        p.stage_us[name] += us; p.stage_calls[name] += 1;
    }
};

class Session {
public:
    Session(const std::string &weights_dir, const std::string &kernels_dir, int max_batch)
        : max_batch_(max_batch) {
        try {
            if (max_batch < 1 || max_batch > MAX_BATCH)
                throw std::invalid_argument("max_batch must be 1.." + std::to_string(MAX_BATCH));
            HIP_CHECK(hipInit(0));
            HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
            profiler.stream = stream_;

            auto spans16 = read_manifest(weights_dir + "/manifest_f16.txt");
            auto spans32 = read_manifest(weights_dir + "/manifest.txt");
            auto blob16 = read_file(weights_dir + "/weights_f16.bin");
            auto blob32 = read_file(weights_dir + "/weights.bin");
            check_spans(spans16, blob16.size(), 2, "manifest_f16.txt");
            check_spans(spans32, blob32.size(), 4, "manifest.txt");
            HIP_CHECK(hipMalloc(&weights16_, blob16.size()));
            HIP_CHECK(hipMalloc(&weights32_, blob32.size()));
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights16_, blob16.data(), blob16.size()));
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights32_, blob32.data(), blob32.size()));

            kernels_.resize(ARCFACE_KERNEL_COUNT);
            std::vector<bool> loaded(ARCFACE_KERNEL_COUNT, false);
            for (int i = 0; i < ARCFACE_LAUNCH_COUNT; ++i) {
                const auto &l = arcface_launches[i];
                if (!loaded[l.kernel]) {
                    kernels_[l.kernel].load(kernels_dir + "/" + arcface_kernel_stems[l.kernel] + ".hsaco",
                                            arcface_kernel_symbols[l.kernel]);
                    loaded[l.kernel] = true;
                }
                if (l.kind == KIND_CONV || l.kind == KIND_HEAD_MATMUL)
                    weight_off_[i] = require(spans16, l.name, size_t(l.n_size) * l.k_size, "manifest_f16.txt");
                if (l.kind != KIND_CONVERT)
                    bias_off_[i] = require(spans32, std::string(l.name) + "_b", size_t(l.bias_rows) * l.n_size, "manifest.txt");
                if (l.aux & AUX_SLOPE)
                    slope_off_[i] = require(spans32, std::string(l.name) + "_slope", size_t(l.n_size), "manifest.txt");
            }

            buffers_.resize(ARCFACE_BUFFER_COUNT);
            for (int b = 0; b < ARCFACE_BUFFER_COUNT; ++b)
                HIP_CHECK(hipMalloc(&buffers_[b], arcface_buffer_bytes[b] * max_batch));
            HIP_CHECK(hipMalloc(&input_, input_bytes(max_batch)));
            // The embeddings come back to pinned host memory: the copy is faster.
            HIP_CHECK(hipHostMalloc(&output_host_, output_elements(max_batch) * sizeof(float), hipHostMallocDefault));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Session() { release(); }

    int max_batch() const { return max_batch_; }
    Profiler profiler;

    static size_t input_bytes(int batch) { return size_t(batch) * SIZE * SIZE * CHANNELS; }
    static size_t output_elements(int batch) { return size_t(batch) * ARCFACE_EMBEDDING; }
    // The last run's embeddings, [batch][512] f32.
    const float *output() const { return output_host_; }

    void check_batch(int batch) const {
        if (batch < 1 || batch > max_batch_)
            throw std::invalid_argument("batch must be 1.." + std::to_string(max_batch_) +
                                        " (max_batch at creation), got " + std::to_string(batch));
    }

    void check_input(const uint8_t *input, size_t bytes, int batch) const {
        if (!input) throw std::invalid_argument("input must not be null");
        check_batch(batch);
        const size_t want = input_bytes(batch);
        if (bytes != want)
            throw std::invalid_argument("input has " + std::to_string(bytes) + " bytes; batch " +
                                        std::to_string(batch) + " requires exactly " + std::to_string(want) +
                                        " (" + std::to_string(SIZE) + "x" + std::to_string(SIZE) + " BGR uint8 per image)");
    }

    void forward(int batch) {
        for (int i = 0; i < ARCFACE_LAUNCH_COUNT; ++i) {
            const auto &l = arcface_launches[i];
            Stage stage(profiler, l.stage);
            KernArgs args;
            void *src = l.src_buf < 0 ? input_ : buffers_[l.src_buf];
            void *dst = buffers_[l.dst_buf];
            unsigned gx, gy, gz = 1;
            if (l.kind == KIND_CONVERT) {            // one workgroup per image row
                args.scalar_i32(batch * l.h);
                args.pointer(src); args.pointer(dst);
                gx = batch * l.h; gy = 1;
            } else if (l.kind == KIND_HEAD_REDUCE) { // one workgroup per image
                args.scalar_i32(batch);
                args.pointer(src);
                args.pointer((char *)weights32_ + bias_off_[i] * 4);
                args.pointer(dst);
                gx = batch; gy = 1;
            } else {                                 // conv3x3 / head matmul: 64-row M tiles
                const int m = l.kind == KIND_CONV ? batch * l.ho * l.wo : batch;
                args.scalar_i32(m);
                args.pointer(src);
                args.pointer((char *)weights16_ + weight_off_[i] * 2);
                args.pointer((char *)weights32_ + bias_off_[i] * 4);
                args.pointer(dst);
                if (l.aux & AUX_RESIDUAL) args.pointer(buffers_[l.extra_buf]);
                if (l.aux & AUX_SLOPE) args.pointer((char *)weights32_ + slope_off_[i] * 4);
                gx = l.n_size / (l.kind == KIND_CONV ? l.tile : TILE);
                gy = (m + TILE - 1) / TILE;
                gz = l.kind == KIND_HEAD_MATMUL ? l.splits : 1;
            }
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
            HIP_CHECK(hipModuleLaunchKernel(kernels_[l.kernel].function, gx, gy, gz, THREADS, 1, 1, 0,
                                            stream_, nullptr, config));
        }
    }

    void synchronize() { HIP_CHECK(hipStreamSynchronize(stream_)); }

    void download(int batch) {
        Stage stage(profiler, "download embeddings");
        HIP_CHECK(hipMemcpyAsync(output_host_, buffers_[ARCFACE_OUTPUT_BUFFER],
                                 output_elements(batch) * sizeof(float), hipMemcpyDeviceToHost, stream_));
    }

    // One call of the ABI: validate everything before touching the GPU.
    void session_run(const uint8_t *input, size_t bytes, int batch, float *embeddings, size_t embeddings_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) throw std::runtime_error("session is unusable after failed GPU recovery; create a new session");
        check_input(input, bytes, batch);
        if (!embeddings) throw std::invalid_argument("embeddings must not be null");
        if (embeddings_elements != output_elements(batch))
            throw std::invalid_argument("embeddings has " + std::to_string(embeddings_elements) +
                                        " f32 elements; batch " + std::to_string(batch) + " requires exactly " +
                                        std::to_string(output_elements(batch)) + " (batch * " +
                                        std::to_string(ARCFACE_EMBEDDING) + ")");
        try {
            HIP_CHECK(hipMemcpyAsync(input_, input, bytes, hipMemcpyHostToDevice, stream_));
            forward(batch);
            download(batch);
            synchronize();
        } catch (...) {
            // A launch can fail after an asynchronous upload was accepted. Drain
            // the stream while holding the session lock, before the caller can
            // reuse its input (which may be pinned memory). Preserve the original
            // exception; a failed drain also permanently disables further runs.
            if (hipStreamSynchronize(stream_) != hipSuccess) failed_ = true;
            throw;
        }
        memcpy(embeddings, output_host_, output_elements(batch) * sizeof(float));
    }

    void print_profile(int forwards, int batch) const {
        double total = 0;
        for (const auto &e : profiler.stage_us) total += e.second;
        std::vector<std::pair<double, std::string>> rows;
        for (const auto &e : profiler.stage_us) rows.push_back({e.second, e.first});
        std::sort(rows.rbegin(), rows.rend());
        printf("stage breakdown over %d forward pass(es), %d image(s):\n", forwards, forwards * batch);
        for (const auto &r : rows)
            printf("  %-28s %9.3f ms  %5.1f%%  (%d launches)\n", r.second.c_str(), r.first / 1000.0,
                   total ? 100.0 * r.first / total : 0.0, profiler.stage_calls.at(r.second));
        printf("  %-28s %9.3f ms\n", "total", total / 1000.0);
    }

private:
    void release() noexcept {
        // Construction can fail after any individual allocation or module load.
        // Clear every handle as it is released so this is also safe for normal
        // destruction and for partially initialized vectors.
        if (stream_) (void)hipStreamSynchronize(stream_);
        for (auto *&b : buffers_) {
            if (b) (void)hipFree(b);
            b = nullptr;
        }
        if (output_host_) (void)hipHostFree(output_host_);
        if (input_) (void)hipFree(input_);
        if (weights16_) (void)hipFree(weights16_);
        if (weights32_) (void)hipFree(weights32_);
        output_host_ = nullptr;
        input_ = weights16_ = weights32_ = nullptr;
        for (auto &k : kernels_) {
            if (k.module) (void)hipModuleUnload(k.module);
            k.module = nullptr;
            k.function = nullptr;
        }
        if (stream_) (void)hipStreamDestroy(stream_);
        stream_ = nullptr;
        profiler.stream = nullptr;
    }

    int max_batch_;
    bool failed_ = false;
    std::mutex mutex_;
    std::vector<Kernel> kernels_;
    std::vector<void *> buffers_;
    hipStream_t stream_ = nullptr;
    void *input_ = nullptr, *weights16_ = nullptr, *weights32_ = nullptr;
    float *output_host_ = nullptr;
    std::map<int, size_t> weight_off_, bias_off_, slope_off_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (!error || capacity == 0) return;
    std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

void clear_error(char *error, size_t capacity) noexcept {
    if (error && capacity) error[0] = '\0';
}

}  // namespace

struct arcface_session {
    Session value;
    arcface_session(const char *w, const char *k, int b) : value(w, k, b) {}
};

extern "C" uint32_t arcface_abi_version(void) { return ARCFACE_ABI_VERSION; }
extern "C" int arcface_input_size(void) { return SIZE; }
extern "C" int arcface_embedding_size(void) { return ARCFACE_EMBEDDING; }
extern "C" int arcface_max_batch(const arcface_session *s) { return s ? s->value.max_batch() : 0; }
extern "C" void arcface_destroy(arcface_session *s) { delete s; }

extern "C" int arcface_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                              arcface_session **out_session, char *error, size_t error_capacity) {
    clear_error(error, error_capacity);
    if (!out_session) { write_error(error, error_capacity, "out_session must not be null"); return ARCFACE_INVALID_ARGUMENT; }
    *out_session = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out_session = new arcface_session(weights_dir, kernels_dir, max_batch);
        return ARCFACE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, error_capacity, e.what()); return ARCFACE_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { write_error(error, error_capacity, e.what()); return ARCFACE_ERROR; }
      catch (...)                            { write_error(error, error_capacity, "unknown C++ exception"); return ARCFACE_ERROR; }
}

extern "C" int arcface_run(arcface_session *s, const uint8_t *input, size_t input_bytes, int batch,
                           float *embeddings, size_t embeddings_elements, char *error, size_t error_capacity) {
    clear_error(error, error_capacity);
    try {
        if (!s) throw std::invalid_argument("session must not be null");
        s->value.session_run(input, input_bytes, batch, embeddings, embeddings_elements);
        return ARCFACE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, error_capacity, e.what()); return ARCFACE_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { write_error(error, error_capacity, e.what()); return ARCFACE_ERROR; }
      catch (...)                            { write_error(error, error_capacity, "unknown C++ exception"); return ARCFACE_ERROR; }
}

#ifndef ARCFACE_LIBRARY
namespace {

int parse_integer(const std::string &option, const std::string &text) {
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || value < INT_MIN || value > INT_MAX)
        throw std::invalid_argument(option + " must be an integer, got '" + text + "'");
    return static_cast<int>(value);
}

// CLI: the same session driven from files, for validation and profiling.
//   host/arcface --weights build/weights --kernels build/kernels --input crops.bin
//                [--batch N] [--repeat N] [--profile] [--output embeddings.bin]
// --input holds either one 112x112x3 BGR uint8 crop, replicated across the
// batch, or exactly --batch of them. Each timed run is exactly what the ABI does
// (upload, forward, download). --output writes the [batch][512] f32 embeddings,
// for the comparison against onnxruntime.
int cli_main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels", input_path, output_path;
    int batch = 1, repeat = 1; bool profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--weights") weights_dir = next();
        else if (a == "--kernels") kernels_dir = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output") output_path = next();
        else if (a == "--batch") batch = parse_integer(a, next());
        else if (a == "--repeat") repeat = parse_integer(a, next());
        else if (a == "--profile") profile = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    if (batch < 1 || batch > MAX_BATCH) throw std::invalid_argument("--batch must be 1.." + std::to_string(MAX_BATCH));
    if (repeat < 1) throw std::invalid_argument("--repeat must be at least 1");
    if (input_path.empty())
        throw std::invalid_argument("--input is required (" + std::to_string(SIZE) + "x" + std::to_string(SIZE) + "x3 BGR uint8 per crop)");
    const size_t per_image = Session::input_bytes(1);
    std::vector<char> raw;
    try { raw = read_file(input_path); } catch (const std::exception &e) { throw std::invalid_argument(e.what()); }
    if (raw.empty() || raw.size() % per_image != 0)
        throw std::invalid_argument(input_path + " is " + std::to_string(raw.size()) + " bytes, not a multiple of one " +
                                    std::to_string(SIZE) + "x" + std::to_string(SIZE) + "x3 uint8 crop");
    const size_t supplied = raw.size() / per_image;
    if (supplied != 1 && supplied != size_t(batch))
        throw std::invalid_argument(input_path + " holds " + std::to_string(supplied) + " crops; expected 1 (replicated) or " +
                                    std::to_string(batch) + " (--batch)");
    std::vector<uint8_t> input(per_image * batch);
    for (int b = 0; b < batch; ++b)
        memcpy(input.data() + size_t(b) * per_image, raw.data() + (supplied == 1 ? 0 : size_t(b) * per_image), per_image);

    Session session(weights_dir, kernels_dir, batch);
    std::vector<float> embeddings(Session::output_elements(batch));
    auto run_once = [&]() {
        session.session_run(input.data(), input.size(), batch, embeddings.data(), embeddings.size());
    };
    run_once();                                            // warm
    session.profiler.enabled = profile;
    session.profiler.stage_us.clear(); session.profiler.stage_calls.clear();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_once();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int images = repeat * batch;
    printf("{\"batch\": %d, \"images\": %d, \"total_ms\": %.3f, \"ms_per_image\": %.4f, \"img_per_s\": %.2f}\n",
           batch, images, ms, ms / images, images / (ms / 1000.0));
    if (profile) session.print_profile(repeat, batch);
    if (!output_path.empty()) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + output_path);
        out.write(reinterpret_cast<const char *>(embeddings.data()),
                  static_cast<std::streamsize>(embeddings.size() * sizeof(float)));
        if (!out) throw std::runtime_error("cannot write all of " + output_path);
    }
    return ARCFACE_OK;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return cli_main(argc, argv);
    } catch (const std::invalid_argument &e) { fprintf(stderr, "%s\n", e.what()); return ARCFACE_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { fprintf(stderr, "%s\n", e.what()); return ARCFACE_ERROR; }
      catch (...)                            { fprintf(stderr, "unknown C++ exception\n"); return ARCFACE_ERROR; }
}
#endif
