"""im2col_f16 vs an explicit NumPy im2col, at tiny shapes that exercise every
halo case, the odd 7x7 stage at stride 1, the 14->7 stride-2 conv, and real
layer shapes with a batch.

Exact comparison: the kernel only moves f16 values or writes zeros.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

P = "arcface.im2col_f16"


def im2col_ref(x: np.ndarray, stride: int, k_pad: int) -> np.ndarray:
    """x [B,H,W,Cp] -> cols [B*Ho*Wo, k_pad], k = (dy*3+dx)*Cp + c, halo and tail zero."""
    b, h, w, cp = x.shape
    ho, wo = h // stride, w // stride
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = np.zeros((b, ho, wo, 9, cp), x.dtype)
    for dy in range(3):
        for dx in range(3):
            cols[:, :, :, dy * 3 + dx, :] = xp[:, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride, :]
    cols = cols.reshape(b * ho * wo, 9 * cp)
    return np.concatenate([cols, np.zeros((cols.shape[0], k_pad - 9 * cp), x.dtype)], axis=1)


def main() -> int:
    ok = True
    rng = np.random.default_rng(5)
    with workdir() as tmp:
        tmp = Path(tmp)
        # (name, B, H, W, cin_pad, storage stride, conv stride)
        for name, batch, h, w, cp, cs, stride in (("tiny s1", 2, 4, 6, 8, 8, 1), ("tiny s2", 2, 4, 6, 8, 8, 2),
                                                 ("odd 3x5 s1", 3, 3, 5, 8, 8, 1), ("odd 7x7 s1", 3, 7, 7, 64, 64, 1),
                                                 ("14->7 s2", 2, 14, 14, 64, 64, 2),
                                                 ("stem 112^2", 2, 112, 112, 8, 8, 1), ("56^2 s2", 2, 56, 56, 64, 64, 2),
                                                 ("28^2 s1", 2, 28, 28, 128, 128, 1)):
            k_pad = (9 * cp + 31) // 32 * 32
            hsaco = tmp / f"im2col_{h}_{w}_{cp}_{stride}.hsaco"
            compile_kernel(ROOT / "kernels/im2col_f16.loom", "arcface_im2col_f16",
                           {f"{P}.height": h, f"{P}.width": w, f"{P}.stride": stride,
                            f"{P}.cin_pad": cp, f"{P}.cin_stride": cs, f"{P}.k_pad": k_pad}, hsaco)
            x = rng.standard_normal((batch, h, w, cs)).astype(np.float16)
            ho, wo = h // stride, w // stride
            m = batch * ho * wo
            (cols,), timing = launch(hsaco, "arcface_im2col_f16", (batch * ho, 1, 1), (256, 1, 1),
                                     [("i32", m), ("in_f16", x.reshape(-1, cs)),
                                      ("out_f16", ((m, k_pad), np.float16))], tmp, repeat=10)
            want = im2col_ref(x[..., :cp], stride, k_pad)
            ok &= report(f"{name:<11s} B={batch} {h}x{w}x{cp} -> [{m}x{k_pad}] "
                         f"({timing['per_launch_us']:8.2f} us, {m * k_pad * 2 / (timing['per_launch_us'] * 1e-6) / 1e9:5.1f} GB/s out)",
                         cols, want, atol=0, rtol=0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
