# arcface-loom

insightface's `w600k_r50` face recogniser (ArcFace, an iResNet-50) written in
**Loom**, AMD's kernel language from [ROCm/hrx-system](https://github.com/ROCm/hrx-system),
for the Radeon 8060S (gfx1151) in a Strix Halo APU. The third sibling of
[dinov3-loom](https://github.com/zacharydenton/dinov3-loom) and
[scrfd-loom](https://github.com/zacharydenton/scrfd-loom), and it reuses their
convolution: with scrfd-loom it runs the whole `buffalo_l` face pipeline --
detect, align, embed -- on Loom.

It is a drop-in for insightface's `ArcFaceONNX`: the same alignment (insightface's
`norm_crop`, vendored), the same normalisation, only the network replaced. On the
six faces of the sample image its 512-d embeddings agree with onnxruntime to
cosine 0.99999 and with insightface's own embeddings to 0.99998, and the 6x6
face-similarity matrix -- what recognition actually uses -- matches insightface's
to 0.0002.

## Why

The deployed model is an ONNX file, and the way to run it on this iGPU is
onnxruntime's MIGraphX provider. The question the siblings ask is whether that
provider's number is the silicon or the toolchain; Loom lets you write the
convolution directly, so the way to find out is to write it. This repo answers it
for the recogniser, reusing scrfd-loom's implicit-GEMM 3x3 conv unchanged and
adding only what ArcFace needs.

## Model

`w600k_r50.onnx` (glint360k R50) from insightface's `buffalo_l` pack: input a
112x112 aligned face crop, output an un-normalised 512-d embedding. An iResNet-50:
a stem, 24 residual blocks in four stages (planes 112/56/28/14/7, widths
64/128/256/512, [3,4,14,3] blocks) of `BN -> Conv3x3 -> PReLU -> Conv3x3 ->
Add(shortcut)`, then `BN -> Flatten -> Gemm(25088->512) -> BN`. 43.6 M
parameters, 87 MB in f16, 25.7 MB of it in the final fully-connected layer.
PReLU is the only nonlinearity; there is no pooling.

## Status

Complete, validated, benchmarked. One inference path, ten kernel sources, all of
them scrfd-loom's or dinov3-loom's with the epilogue changed:

| kernel | what |
| --- | --- |
| `conv3x3_f16_wmma` (+ the `n128` 64x128-tile family) with `prelu`, `bnprelu`, `add` epilogues | scrfd-loom's implicit-GEMM 3x3 conv -- the WMMA matmul with its A-staging load replaced by an 8-wide im2col gather that runs one step ahead of the multiply, so the `[M][K]` matrix never exists. 49 of the 53 convs. `prelu` is `max(x,0) + slope[n]*min(x,0)`; `bnprelu` folds the preceding BatchNorm into the weights and a 9-entry border-bias table read per output pixel; `add` folds the residual |
| `matmul_splitk_f16_wmma` + `splitk_reduce_f32` | the 25088->512 head, M = batch. dinov3-loom's split-K matmul, K widened past 8192 and the split count a config, so a handful of 64x64 tiles become hundreds of workgroups across the 40 CUs; the reduce sums the partials and adds the bias in f32 |
| `hwc_u8_to_nhwc_f16` | the aligned BGR uint8 crop to normalised RGB NHWC f16: insightface's `(x-127.5)/127.5` with swapRB in one pass, so the host uploads bytes, never a blob |
| `im2col_f16` | the explicit gather -- the reference for the implicit one, and the permanent fallback |

Every BatchNorm, PReLU and Add is folded away: **56 launches per forward pass**,
from 130 ONNX nodes, four buffers, 3.8 MB per image. `tools/gen_launch_table.py`
generates that schedule, the buffer assignment and the kernel build list from the
ONNX file; nothing about the network is transcribed by hand. Activations are NHWC
f16, weights f16, accumulation f32 throughout.

The 26 BatchNorms cannot be folded into the following conv by the usual
weights-and-bias trick, because several have per-channel scales near 1e-26 (dead
channels) that would need bias corrections of 1e32. The block BNs fold into the
next conv's weights plus a **9-entry border-bias table** -- one bias per
(top/mid/bottom x left/mid/right) output-pixel class, because the conv zero-pads
*after* the BatchNorm so a border pixel misses the shift of its outside taps. The
head's two BatchNorms fold exactly into the Gemm's columns and rows. All three
folds are checked in float64 against the unfolded graph (`tools/test_export_fold.py`).

## Correctness

`tools/reference.py` is a float64 NumPy interpreter of the ONNX graph, agreeing
with onnxruntime to 5e-7 of the output's range. Every kernel is graded against it,
not against onnxruntime, so a kernel bug cannot hide behind a matching bug in the
harness. End to end (`tools/validate.py`), the 512-d embedding of each of the six
faces of the sample image:

```
  PASS face 0 vs onnxruntime: cosine=0.9999993
  ...
  PASS vs insightface's own embeddings: cosine min=0.9999979
  PASS 6x6 similarity matrix vs insightface: max |delta| = 0.0002
  PASS batch 6 equals 6 batch-1 runs bit for bit
```

The insightface check is against a fixture of insightface's *own*
`ArcFaceONNX.get()` embeddings, captured once with the real package, so the
vendored alignment and the whole recogniser are graded against production. The
alignment reproduces insightface's `norm_crop` -- a scikit-image Umeyama fit,
which insightface runs in float32, so the crop is bit-identical under the BLAS
that captured the fixture and within a pixel level under another; the embeddings
are cosine 0.99998 either way (`docs/notes.md`).

## Benchmark

`tools/benchmark.py` times what a caller pays after alignment: `ArcFaceLoom.get_feat`
on aligned crops (upload, the network, download), interleaved with onnxruntime's
MIGraphX provider on the same box, best of three rounds. Unlike scrfd-loom's
detector graph, this one accepts a batch -- it only *declares* `[1, 512]` -- so
MIGraphX is timed batched too, each shape compiled once (about a minute). Measured
on a Radeon 8060S (gfx1151) with other CPU jobs resident, so treat the absolute
numbers as a floor.

| configuration | img/s | ms/img | vs MIGraphX b1 |
| --- | ---: | ---: | ---: |
| arcface-loom `get_feat`, batch 32 | **1658.2** | 0.603 | **9.80x** |
| arcface-loom `get_feat`, batch 16 | 1583.4 | 0.632 | 9.36x |
| arcface-loom `get_feat`, batch 8 | 1491.4 | 0.671 | 8.81x |
| onnxruntime + MIGraphX, batch 16 | 440.3 | 2.271 | 2.60x |
| arcface-loom `get_feat`, batch 1 | 411.8 | 2.428 | 2.43x |
| onnxruntime + MIGraphX, batch 1 | 169.3 | 5.908 | 1.00x |

Loom is ahead of MIGraphX at every matched batch: 2.4x at batch 1, 3.6x at
batch 16, and 9.8x at its own batch 32 over MIGraphX's batch-1 latency. The native
call alone (`host/arcface --repeat`: upload, 56 launches, download) reaches 1626
img/s at batch 32 and 399 at batch 1; `get_feat` adds only the host copies.

Where the time goes at batch 6 (`host/arcface --profile`): the 14x14 stage (256
channels, 14 blocks) is 47% of it, as its 53% share of the FLOPs predicts; the
25088->512 head is 3%. At batch 1 most of the network runs at low occupancy -- the
14x14 stage launches 16 workgroups and the 7x7 stage 8, against ~120 slots on 40
CUs -- and every forward streams the 87 MB of weights from DRAM regardless of
batch, so batched throughput per face is the honest figure. `docs/notes.md`
records the levers.

## Replacing insightface

```python
# before
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
model = ArcFaceONNX("~/.insightface/models/buffalo_l/w600k_r50.onnx")
model.prepare(ctx_id=0)
embedding = model.get(image_bgr, face)          # face.kps from the detector

# after
from arcface_loom import ArcFaceLoom
model = ArcFaceLoom()
embedding = model.get(image_bgr, face)          # same (512,) f32 embedding
```

`get(img, face)` aligns the face from its five landmarks with insightface's
`norm_crop` and embeds the crop; `get_feat(crops)` takes already-aligned
`(112, 112, 3)` uint8 BGR crops (a list or a stacked array) and returns
`(n, 512)`, up to `max_batch` (default 16) per GPU call; `embed(img, kps_array)`
does many faces of one image; `compute_sim` is insightface's cosine. The session
is resident: kernels and weights load once, buffers are sized for `max_batch` at
construction, and each call is one ctypes entry into `build/libarcface.so`. It is
thread-safe, closable, and a context manager, and paths come from
`ARCFACE_LOOM_WEIGHTS`, `ARCFACE_LOOM_KERNELS` and `ARCFACE_LOOM_LIBRARY`.

### What to know before swapping

- **Accuracy is cosine 0.99998 against insightface, not bitwise.** That is well
  inside the noise for verification and clustering; if you compare against a
  gallery built with the ONNX model, the ranking is unchanged, but rebuild the
  gallery rather than mixing the two if you threshold tightly.
- **The shape is fixed at 112x112 -> 512, gfx1151.** The kernels are compiled for
  it; `LOOM_TARGET` changes the chip but nothing else here has been measured on
  another.
- **The embedding is un-normalised**, exactly as `ArcFaceONNX` returns it;
  `compute_sim` and any downstream store normalise it, as insightface does.
- **Construction is the expensive operation.** It initializes HIP, uploads the
  weights and loads the modules once; calls after that are one upload, the
  network, one download. Use `close()` or the context manager to release the
  session; create models after forking, not before.

## Running it

Needs the Loom toolchain from [ROCm/hrx-system](https://github.com/ROCm/hrx-system)
(`scripts/env.sh` points at the build) and ROCm for `hipcc`; `requirements.txt`
for Python; the `w600k_r50.onnx` file from insightface's `buffalo_l` pack.

```console
$ pip install -r requirements.txt
$ pip install -e .
$ source scripts/env.sh
$ python3 tools/export_weights.py           # 54 f16 matrices with the BNs folded in, 90 MB
$ python3 tools/gen_launch_table.py         # schedule + kernel build list from the graph
$ ./scripts/build_kernels.sh                # 24 HSACOs
$ ./scripts/build_host.sh                   # host/arcface CLI + build/libarcface.so
$ ./scripts/test.sh                         # everything
```

`scripts/test.sh` is the one test command: formatting, the generated files against
their generators, the float64 fold identities, the build, every unit test against
float64, the runner's error paths, then the end-to-end comparisons against
onnxruntime and insightface. `--quick` skips the last group, the only part needing
onnxruntime. The fixture of insightface's own embeddings is regenerated with
`tools/capture_fixture.py` (it needs scikit-image, via `uv run`).

## Layout

```
kernels/    the ten .loom sources and their generated epilogue variants
host/       arcface.cpp + arcface.h (resident session, C ABI, CLI), graph_table.inc (generated),
            loomrun.cpp (the single-kernel runner the unit tests use)
tools/      graph.py (the ONNX graph, resolved), reference.py (the float64 oracle),
            export_weights.py (the folds), gen_launch_table.py, gen_conv.py, gen_variants.py,
            capture_fixture.py, tests, benchmark.py
arcface_loom.py         the Python API
arcface_loom_align.py   insightface's norm_crop, vendored
docs/       notes.md -- the graph, the folds, the levers
```

## Licence

Apache-2.0 (`LICENSE`), matching hrx-system. `arcface_loom_align.py` reproduces
code from insightface (MIT) and scikit-image (BSD-3); see `THIRD_PARTY_NOTICES.md`.
