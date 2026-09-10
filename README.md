# arcface-hrx

InsightFace w600k_r50 inference in Rust, using [hrx-rs](https://github.com/zacharydenton/hrx-rs)
for GPU execution and Loom compilation. One Cargo package provides the library
and CLI. Weights, kernels, activation buffers and readback storage stay resident;
HRX graphs are recorded once per encountered batch size and replayed.

Requires Rust 1.88+, Linux x86-64 and a Radeon 8060S (`gfx1151`). HRX 0.4.0
provisions its verified runtime and compiler bundle; its native Linux bundle
requires glibc 2.43 or newer. Model files are supplied separately.

```bash
cargo build --release
```

## Library

Load the original `w600k_r50.onnx` from InsightFace's `buffalo_l` model pack.
`onnx-protobuf` parses ONNX; the importer validates the supported graph and
folds BatchNorm, PReLU and residual operations into the production kernels.
No exported weight blobs or generated launch tables are required.

```rust,no_run
use arcface_hrx::{ArcFace, Options};

# fn main() -> anyhow::Result<()> {
let mut model = ArcFace::load("w600k_r50.onnx", Options::default())?;
let crops = vec![0u8; 112 * 112 * 3];
let embeddings = model.embeddings(&crops)?; // Vec<[f32; 512]>
# Ok(())
# }
```

`embeddings` accepts packed, aligned 112×112 uint8 BGR crops and returns
unnormalized embeddings, matching InsightFace. RGB conversion and normalization
run on the GPU. `embed(image, width, height, landmarks)` accepts a packed BGR
image and a slice of five-point `[[f32; 2]; 5]` landmarks, then aligns and embeds
each face. `alignment::crop` exposes the alignment separately; `similarity`
computes cosine similarity and rejects zero or non-finite embeddings.

`Options` selects a device index and maximum resident batch (default 16, range
1–64). Larger crop batches are chunked; empty batches return empty results.
The four activation buffers use 3.8 MB per resident image, plus weights and
input/readback storage.

Combine with [scrfd-hrx](https://github.com/zacharydenton/scrfd-hrx): pass the
original BGR image and each detection's `landmarks` to `embed`. No conversion
of landmark types is required.

## CLI

```bash
cargo run --release -- --model w600k_r50.onnx \
  --input aligned-crops.bgr --output embeddings.f32
```

Input is packed BGR bytes, with 112×112×3 bytes per crop. Output is little-endian
float32, 512 values per crop. Add `--benchmark 100` for timings; benchmark input
must fit one resident batch.

## Execution and validation

Inference requires `&mut` access to the model. A model owns its stream; use
separate models for independent concurrent callers. GPU failures return errors
and make the session unusable. Drop releases owned resources through HRX.
The fixed production kernels are validated only for `gfx1151`.

Warm calls reuse compiled kernels, device allocations and graphs. Uploads are
queued; readback copies share the inference stream and complete before host
access. Graph dependencies preserve launch order and activation-buffer reuse.
`benchmark` reports alternating graph/direct forward timings, excluding transfers;
the CLI also reports warm end-to-end timing. Both are synchronized host timings,
not hardware timestamp measurements. See [current measurements](docs/benchmark-2026-09-10.md).

```bash
cargo test
cargo clippy --all-targets -- -D warnings
ARCFACE_MODEL=/path/to/w600k_r50.onnx \
  cargo test --release -- --include-ignored --test-threads=1
```

CPU tests run without a GPU or model files. Ignored tests require the model and
hardware; they cover numerical agreement, changing inputs, partial batches and
graph replay. Tests compare the unfused ONNX model through a Rust CPU reference,
plus the captured InsightFace fixture. Embedding cosine must exceed 0.99995,
and pairwise similarity error must remain below 0.001.

The lossless fixture preserves the pixels used for the original reference;
JPEG decoders can produce different pixels. These small fixtures establish
numerical agreement, not accuracy on other face datasets.

## License

Project code is Apache-2.0. Model weights have separate terms and are not
included or downloaded by this crate. See [third-party notices](THIRD_PARTY_NOTICES.md)
for model terms and retained source attribution.
