// C ABI for the resident w600k_r50 (ArcFace iResNet-50) inference session.
//
// The same shape as scrfd-loom's host/scrfd.h and dinov3-loom's host/dinov3.h
// so the three models share one ctypes loader: a session created once, runs
// serialized on it, element counts in the ABI so an undersized caller
// allocation is rejected before the GPU sees it.
#ifndef ARCFACE_LOOM_H
#define ARCFACE_LOOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment when the ABI changes incompatibly. Callers compare before creating.
#define ARCFACE_ABI_VERSION 1u

enum {
    ARCFACE_OK = 0,
    ARCFACE_ERROR = 1,
    ARCFACE_INVALID_ARGUMENT = 64,
};

typedef struct arcface_session arcface_session;

uint32_t arcface_abi_version(void);
int arcface_input_size(void);          // 112
int arcface_embedding_size(void);      // 512

// Creates an independent resident session with buffers sized for up to
// max_batch images. On failure returns non-zero, leaves *out_session null and
// writes a NUL-terminated message to error (when error_capacity is non-zero).
int arcface_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                   arcface_session **out_session, char *error, size_t error_capacity);

// Runs one batch. `input` is batch aligned BGR uint8 face crops, each
// 112 x 112 x 3, exactly what insightface's norm_crop builds before
// blobFromImages; the normalisation happens on the GPU. `input_bytes` must be
// batch * 112 * 112 * 3. `embeddings` receives batch rows of 512 f32, the
// un-normalised embedding the ONNX graph outputs (insightface normalises
// downstream); `embeddings_elements` must be exactly batch * 512.
//
// Calls on one session are serialized internally. Destruction must not race a run.
// On a GPU error, queued work is drained before returning. If the drain fails,
// subsequent runs are rejected; destroy the session and create a new one after
// resolving the GPU error. Output is written only after a successful run.
int arcface_run(arcface_session *session, const uint8_t *input, size_t input_bytes, int batch,
                float *embeddings, size_t embeddings_elements, char *error, size_t error_capacity);

int arcface_max_batch(const arcface_session *session);
void arcface_destroy(arcface_session *session);

#ifdef __cplusplus
}
#endif

#endif
