"""The embedding head -- split-K matmul + f32 reduce -- vs float64, on the real
7x7 tensor of a face (row 0) plus random rows, at several split counts; and the
whole head against the graph's own output, so the column permutation and the
two folded BNs are checked on the model's numbers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
import export_weights as E
import test_align as TA

MM = "arcface.matmul_splitk_f16_wmma"
RD = "arcface.splitk_reduce_f32"


def main() -> int:
    ok = True
    rng = np.random.default_rng(33)
    graph = G.load()
    import cv2
    crops = TA.crops(cv2.imread(str(TA.test_image())), TA.fixture())[:1]
    tensors = R.forward(graph, R.blob_from_bgr_f64(crops), keep=True)
    bn_in, fc, bn_out = E.head_ops(graph)
    x = tensors[bn_in.inputs[0]]                                   # [1,512,7,7] f64
    _, c, h, w = x.shape
    w_fc, b_fc = E.fold_head(bn_in, fc, bn_out, h * w, c)
    w16 = w_fc.astype(np.float16)
    b32 = b_fc.astype(np.float32)
    n, k = w16.shape
    want_model = tensors[graph.output][0]

    batch = 5
    a = np.zeros((batch, h * w, c))
    a[0] = x[0].transpose(1, 2, 0).reshape(h * w, c)
    a[1:] = rng.standard_normal((batch - 1, h * w, c)) * np.abs(a[0]).mean()
    a16 = a.reshape(batch, k).astype(np.float16)
    want = a16.astype(np.float64) @ w16.astype(np.float64).T + b_fc

    with workdir() as tmp:
        tmp = Path(tmp)
        for splits in (4, 8, 16, 28, 56, 98, 196):
            assert k % (32 * splits) == 0
            h_mm = tmp / f"splitk_{splits}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_splitk_f16_wmma.loom", "arcface_matmul_splitk_f16_wmma",
                           {f"{MM}.k_size": k, f"{MM}.n_size": n, f"{MM}.splits": splits}, h_mm)
            (partials,), t_mm = launch(h_mm, "arcface_matmul_splitk_f16_wmma",
                                       (n // 64, (batch + 63) // 64, splits), (256, 1, 1),
                                       [("i32", batch), ("in_f16", a16), ("in_f16", w16), ("in", b32),
                                        ("out", ((splits * batch, n), np.float32))], tmp, repeat=10)
            h_rd = tmp / f"reduce_{splits}.hsaco"
            compile_kernel(ROOT / "kernels/splitk_reduce_f32.loom", "arcface_splitk_reduce_f32",
                           {f"{RD}.n_size": n, f"{RD}.splits": splits}, h_rd)
            (out,), t_rd = launch(h_rd, "arcface_splitk_reduce_f32", (batch, 1, 1), (256, 1, 1),
                                  [("i32", batch), ("in", partials), ("in", b32),
                                   ("out", ((batch, n), np.float32))], tmp, repeat=10)
            gbs = k * n * 2 / (t_mm["per_launch_us"] * 1e-6) / 1e9
            ok &= report(f"head K={k} N={n} B={batch} splits={splits:<3d} (matmul {t_mm['per_launch_us']:7.1f} us = "
                         f"{gbs:5.0f} GB/s of weights, reduce {t_rd['per_launch_us']:5.1f} us)",
                         out, want, atol=5e-2, rtol=5e-2)
            if splits == E.head_splits(k):
                cos = float(out[0] @ want_model / (np.linalg.norm(out[0]) * np.linalg.norm(want_model)))
                good = cos > 0.9999
                ok &= good
                print(f"  {'PASS' if good else 'FAIL'} head vs the graph's own embedding (f16 A and W): cosine={cos:.7f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
