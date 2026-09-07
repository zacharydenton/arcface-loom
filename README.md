# arcface-loom

InsightFace's `w600k_r50` face recogniser implemented in
[Loom](https://github.com/ROCm/hrx-system) for the AMD Radeon 8060S (gfx1151).
It converts aligned 112×112 BGR crops into 512-dimensional face embeddings.

The network runs in f16 with f32 accumulation. Alignment and input normalisation
follow InsightFace. Kernels, weights and GPU buffers stay resident between calls.
With [scrfd-loom](https://github.com/zacharydenton/scrfd-loom), it supports the
`buffalo_l` detect → align → embed pipeline.

## Performance and validation

Measured on a Radeon 8060S, including crop upload, inference and embedding
download. Alignment is excluded. Results are the best of three interleaved
rounds with other CPU jobs running.

| Runtime | Batch | Images/s | ms/image |
| --- | ---: | ---: | ---: |
| arcface-loom | 1 | 411.8 | 2.428 |
| ONNX Runtime + MIGraphX | 1 | 169.3 | 5.908 |
| arcface-loom | 16 | 1583.4 | 0.632 |
| ONNX Runtime + MIGraphX | 16 | 440.3 | 2.271 |
| arcface-loom | 32 | 1658.2 | 0.603 |

At matched batch sizes, these measurements show **2.4× at batch 1** and
**3.6× at batch 16**. See [benchmark output](docs/benchmark-2026-09-04.txt)
and [`tools/benchmark.py`](tools/benchmark.py) for the comparison.

On the six faces in InsightFace's sample image, embeddings agree with its
reference embeddings to a minimum cosine similarity of **0.9999979**. The 6×6
similarity matrix differs by at most **0.0002**; batch-6 results equal six
batch-1 calls bit for bit. Tests also compare individual kernels and weight
folds against a float64 NumPy reference.

These checks establish numerical agreement on that fixture. They do not establish
recognition accuracy or ranking preservation for other galleries. Small score
changes can reorder close matches or cross a threshold; validate your own data
before replacing a deployed recogniser.

## Build

Requires Linux x86-64, a gfx1151 GPU, ROCm, Python 3.11+, the Loom compiler, and
`w600k_r50.onnx` from InsightFace's `buffalo_l` pack.

Follow [the build guide](docs/building.md) to install the pinned public Loom
revision and obtain and verify the model. Then, from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/env.sh
python tools/export_weights.py
python tools/gen_launch_table.py
./scripts/build_kernels.sh
./scripts/build_host.sh
./scripts/test.sh
```

`test.sh` rebuilds the assets and runs the full suite. `--quick` skips the final
alignment/reference/end-to-end/API group; it still needs the GPU and model.
CPU ONNX Runtime is sufficient for validation. The benchmark needs a MIGraphX
build of ONNX Runtime.

The Python wheel contains only the loader and alignment code. When using it
outside a checkout, set `ARCFACE_LOOM_WEIGHTS`, `ARCFACE_LOOM_KERNELS` and
`ARCFACE_LOOM_LIBRARY` to the exported weights, compiled kernels and
`libarcface.so`. Build and runtime overrides are listed in the
[build guide](docs/building.md#paths-and-runtime-selection).

## Python API

```python
from arcface_loom import ArcFaceLoom

with ArcFaceLoom(max_batch=16) as model:
    embedding = model.get(image_bgr, face)  # face.kps: five landmarks
    embeddings = model.get_feat(crops)     # (B, 112, 112, 3) uint8 BGR
```

| Method | Input | Result |
| --- | --- | --- |
| `get(image, face)` | BGR image and an InsightFace `Face` or `(5, 2)` landmarks | `(512,)` embedding; also sets `face.embedding` |
| `get_feat(crops)` | One aligned crop, a list, or a stacked uint8 BGR array | `(B, 512)` embeddings, chunked at `max_batch` |
| `embed(image, landmarks)` | BGR image and `(N, 5, 2)` landmarks | `(N, 512)` embeddings |
| `compute_sim(a, b)` | Two embeddings | Cosine similarity |
| `prepare(ctx_id=0)` | InsightFace preparation hook | No-op on the resident session |

Embeddings are unnormalised float32, as returned by `ArcFaceONNX`. Calls on one
model are serialized and thread-safe. Use `close()` or a context manager to
release GPU resources. Create sessions after forking.

To replace recognition in an existing InsightFace `FaceAnalysis` pipeline:

```python
with ArcFaceLoom() as model:
    app.models["recognition"] = model
    app.prepare(ctx_id=0)
    faces = app.get(image_bgr)
```

`prepare(0)` is repeatable. CPU fallback and other device IDs are unsupported.
The ONNX-specific `session`, `model_file` and `forward(batch_data)` interfaces
are not provided. Input size is fixed at 112×112; other GPUs are unvalidated.

## Implementation

The ONNX graph generates the launch schedule and buffer assignments. BatchNorm,
PReLU and residual additions are folded into convolution and head operations,
reducing the 130-node graph to 56 launches. The host uses four activation buffers
(3.8 MB per image) and approximately 90 MB of exported weights.

- [`kernels/`](kernels/): Loom convolution, conversion and matrix kernels.
- [`host/`](host/): resident C ABI, inference CLI and single-kernel test runner.
- [`tools/`](tools/): weight export, code generation, reference, tests and benchmark.
- [Engineering notes](docs/notes.md): weight folds, kernel ABI and performance details.

## License

The code is [Apache-2.0](LICENSE), with MIT and BSD-3-Clause alignment code
covered by [third-party notices](THIRD_PARTY_NOTICES.md).

Model weights have separate terms. InsightFace distributes `buffalo_l` for
non-commercial research; other uses require appropriate model licensing.
Neither the ONNX model nor exported weights are included. See
[InsightFace's license policy](https://github.com/deepinsight/insightface#license).
