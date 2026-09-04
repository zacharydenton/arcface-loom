# Notes

Engineering log for arcface-loom. The rules inherited from the siblings still hold:
`config.get` values are specialization constants (one HSACO per shape), never put
`where [range(...)]` on a launch argument, and every kernel is graded against the float64
reference, not against onnxruntime. See dinov3-loom's and scrfd-loom's `docs/notes.md`.

## The graph

`w600k_r50.onnx`: opset 11, input `[N,3,112,112]`, output declared `[1,512]`. 130 nodes:
53 Conv (45 3x3 s1 p1, 4 3x3 s2 p1, 4 1x1 s2 shortcuts), 26 BatchNormalization, 25 PRelu
(per channel), 24 Add, 1 Flatten, 1 Gemm (`[512, 25088]`). 43.6 M parameters, 87 MB in
f16, 25.7 MB of it in the fc. Stem `Conv -> PRelu -> BN`, then 24 blocks ([3,4,14,3];
planes 112/56/28/14/7; widths 64/128/256/512) of `BN(x) -> Conv3x3 -> PRelu ->
Conv3x3(s1|s2) -> Add(x or Conv1x1s2(x))`, then `BN -> Flatten -> Gemm -> BN`. The
shortcut reads `x` before the BN; every BN's output has exactly one consumer.

onnxruntime runs the graph batched despite the `[1,512]` declaration (a one-line warning;
`log_severity_level = 3` silences it), CPU and MIGraphX alike; MIGraphX compiles once per
batch shape (100-150 s each). Batch-6 output equals six batch-1 runs bit for bit.

## The oracle

`tools/reference.py`, float64 NCHW, agrees with onnxruntime CPU to 5e-7 of the output's
range on the six faces of t1.jpg (16 s for the six).

## Alignment

insightface's `norm_crop` is a Umeyama similarity fit (scikit-image's `_umeyama`) from
the five landmarks to `arcface_dst`, then `cv2.warpAffine`. The landmarks are float32, and
scikit-image keeps the caller's dtype, so the SVD runs in float32 and the resulting 2x3
affine depends on the BLAS: `arcface_loom_align.umeyama` reproduces the fixture bit for
bit under the numpy that captured it (OpenBLAS in a uv environment) and to 9e-5 under the
system numpy (reference LAPACK). Three of the six crops then differ by an intensity level
in a few pixels. The same graph on those crops gives embeddings within 1e-6 cosine of the
fixture's, so the gate is cosine, not crop bytes. Computing in float64 instead would be
more accurate and *less* faithful; the contract is insightface's arithmetic.

## Export

Three folds, all exact in float64 (`tools/test_export_fold.py`, 1e-14 on real
activations):

- **BN -> Conv3x3 s1 p1.** Scale into the weights' input channels. The shift cannot go into
  the bias because the conv zero-pads *after* the BN: a border pixel's outside taps
  contribute nothing, an interior pixel's contribute `sum_c W[n,c,tap] * shift[c]` per tap.
  So the bias is a `[9][Cout]` table by border class (top/mid/bottom x left/mid/right),
  picked per output row in the epilogue. Zero hot-loop cost, no extra tensors. The
  alternative -- applying the BN in the gather -- would sit on the load-to-MMA path the
  register prefetch keeps short; a dual-output epilogue on the producer would add a tensor
  per block. Padding the input with `-shift/scale` is impossible: several BN scales are
  ~1e-26 (dead channels), giving pad values of 1e32.
- **Conv1x1 s2 -> centre tap of a 3x3 s2.** Reads (2yo, 2xo), never outside. 9x the MMA work
  on 4.6% of the FLOPs, and one conv kernel runs every conv. A dedicated stride-2 matmul is
  a lever for later.
- **BN -> Flatten -> Gemm -> BN.** Columns and rows; the columns permuted from NCHW flatten
  order (`c*49 + p`) to the NHWC tensor the kernels hold (`p*512 + c`).

f16 rounding of the folded weights: the dead channels' scaled weights underflow to zero,
which is correct to 1e-26 of the activation.
