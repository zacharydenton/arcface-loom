"""The export's folds against the unfolded graph, in float64, on real activations.

Before any kernel exists, this pins the algebra: a BN followed by a zero-padded
3x3 conv equals the scaled conv plus the border-class bias table; a 1x1 stride-2
conv equals its centre-tap 3x3 stride-2 layout; the head's BN -> Flatten -> Gemm
-> BN equals the permuted, folded matmul on the NHWC tensor.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import reference as R
import export_weights as E
import test_align as TA


def check(name: str, got: np.ndarray, want: np.ndarray, tol: float = 1e-10) -> bool:
    err = np.abs(got - want).max() / (np.abs(want).max() + 1e-300)
    print(f"  {'PASS' if err < tol else 'FAIL'} {name}: rel_err={err:.2e}")
    return err < tol


def main() -> int:
    import cv2
    ok = True
    graph = G.load()
    roles = E.conv_roles(graph)
    img = cv2.imread(str(TA.test_image()))
    crops = TA.crops(img, TA.fixture())[:2]
    tensors = R.forward(graph, R.blob_from_bgr_f64(crops), keep=True)

    # Every BN -> conv pair, on the real activation feeding the BN.
    for op in graph.convs:
        role = roles[op.name]
        if role["variant"] != "bnprelu":
            continue
        bn = role["bn"]
        x = tensors[bn.inputs[0]]
        want = R.conv2d(R.batchnorm(x, bn.scale, bn.shift), op.weight, op.bias, 1, 1)
        w, table = E.fold_bn_conv(bn.scale, bn.shift, op.weight, op.bias)
        n, cout, ho, wo = want.shape
        cls = E.border_class(ho, wo)                                   # [ho*wo]
        got = R.conv2d(x, w, np.zeros(cout), 1, 1)
        got += table[cls].T.reshape(1, cout, ho, wo)
        ok &= check(f"{op.name} bn fold ({ho}x{wo})", got, want)

    # The stride-2 shortcuts as centre-tap 3x3 stride-2 convs.
    for op in graph.convs:
        if op.ksize == 1:
            x = tensors[op.inputs[0]]
            want = R.conv2d(x, op.weight, op.bias, 2, 0)
            got = R.conv2d(x, E.centre_tap(op.weight), op.bias, 2, 1)
            ok &= check(f"{op.name} centre-tap shortcut", got, want)

    # The head on the real 7x7 tensor, NHWC.
    bn_in, fc, bn_out = E.head_ops(graph)
    x = tensors[bn_in.inputs[0]]
    want = tensors[graph.output]
    _, c, h, w_ = x.shape
    w_fc, b_fc = E.fold_head(bn_in, fc, bn_out, h * w_, c)
    a = x.transpose(0, 2, 3, 1).reshape(x.shape[0], -1)              # NHWC rows, p*C + c
    got = a @ w_fc.T + b_fc
    ok &= check("head fold (bn -> flatten -> gemm -> bn)", got, want)

    # The packed gather order reproduces the conv (one real 3x3 layer, unpadded K).
    op = graph.convs[1]
    x = tensors[op.inputs[0]]
    w16, info = E.pack(op.weight)
    want = R.conv2d(x, op.weight.astype(np.float16).astype(np.float64), op.bias, 1, 1)
    n, cin, hh, ww = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.zeros((n, hh, ww, 9, info["cin_pad"]))
    for dy in range(3):
        for dx in range(3):
            cols[:, :, :, dy * 3 + dx, :cin] = xp[:, :, dy:dy + hh, dx:dx + ww].transpose(0, 2, 3, 1)
    cols = cols.reshape(n * hh * ww, 9 * info["cin_pad"])
    got = cols @ w16.astype(np.float64)[:, :cols.shape[1]].T + np.pad(op.bias, (0, info["cout_pad"] - op.cout))
    got = got.reshape(n, hh, ww, -1)[..., :op.cout].transpose(0, 3, 1, 2)
    ok &= check(f"{op.name} packed gather order", got, want)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
