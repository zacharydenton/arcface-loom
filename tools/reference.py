"""Float64 NumPy interpreter for the w600k_r50 graph, in NCHW, matching onnxruntime.

Deliberately independent of every kernel and of the export's folds and NHWC
layout: this is the oracle the Loom kernels are graded against, so a mistake
in the export cannot be mirrored here. BatchNormalization is applied where the
graph applies it (before the conv's zero padding), convolution is explicit
im2col + matmul in float64, which is slow (a second or two per image) and
exactly right.
"""
from __future__ import annotations

import numpy as np

import graph as G


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """x [N,C,H,W] f64, w [Cout,Cin,k,k], b [Cout] -> [N,Cout,Ho,Wo] f64."""
    n, c, h, wd = x.shape
    cout, cin, k, _ = w.shape
    assert cin == c
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    # im2col: [N, Ho, Wo, C, k, k] via stride tricks, then one matmul per image
    sN, sC, sH, sW = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(n, ho, wo, c, k, k),
        strides=(sN, sH * stride, sW * stride, sC, sH, sW), writeable=False)
    cols = cols.reshape(n, ho * wo, c * k * k)
    out = cols @ w.reshape(cout, -1).T.astype(np.float64) + b.astype(np.float64)
    return out.reshape(n, ho, wo, cout).transpose(0, 3, 1, 2)


def batchnorm(x: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Per-channel affine on axis 1, for [N,C,H,W] and [N,C] alike."""
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return x * scale.reshape(shape) + shift.reshape(shape)


def prelu(x: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + slope.reshape(1, -1, 1, 1) * np.minimum(x, 0.0)


def forward(graph: G.Graph, pixel_values: np.ndarray, keep: bool = False):
    """pixel_values [N,3,112,112] (already (x-127.5)/127.5, RGB) -> [N,512] embedding.

    With keep=True returns every tensor by ONNX name instead (for the kernel tests,
    which need real activations)."""
    t: dict[str, np.ndarray] = {graph.input: np.asarray(pixel_values, dtype=np.float64)}
    for op in graph.ops:
        a = t[op.inputs[0]]
        if op.kind == "conv":
            t[op.output] = conv2d(a, op.weight, op.bias, op.stride, op.pad)
        elif op.kind == "bn":
            t[op.output] = batchnorm(a, op.scale, op.shift)
        elif op.kind == "prelu":
            t[op.output] = prelu(a, op.slope)
        elif op.kind == "add":
            t[op.output] = a + t[op.inputs[1]]
        elif op.kind == "flatten":
            t[op.output] = a.reshape(a.shape[0], -1)
        elif op.kind == "gemm":
            t[op.output] = a @ op.weight.T.astype(np.float64) + op.bias.astype(np.float64)
        else:
            raise NotImplementedError(op.kind)
    return t if keep else t[graph.output]


def blob_from_bgr_f64(crops_bgr_u8: np.ndarray) -> np.ndarray:
    """[N,112,112,3] BGR uint8 -> [N,3,112,112] float64 RGB with insightface's (x-127.5)/127.5."""
    x = np.asarray(crops_bgr_u8).astype(np.float64)[..., ::-1].transpose(0, 3, 1, 2)
    return (x - 127.5) / 127.5
