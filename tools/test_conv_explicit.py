"""Day-one convolution: im2col_f16 + the WMMA matmul, vs the float64 reference.

This chains the two kernels on real layer shapes with the real exported weight
layout and grades the result against reference.conv2d in NCHW, so a mistake in
the gather order or the weight layout shows up here, not in the model. It covers
the stem, a folded-BN conv1 (the border-bias table added in NumPy), a stride-2
conv2, a centre-tap shortcut, the 14x14 stage and the odd 7x7 stage.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
import export_weights as E

IM = "arcface.im2col_f16"
MM = "dinov3.matmul_bias_f16_wmma_af16_cf16"


def main() -> int:
    ok = True
    rng = np.random.default_rng(9)
    graph = G.load()
    convs = {op.name: op for op in graph.convs}
    roles = E.conv_roles(graph)
    cases = [("c00", 1), ("c01", 2), ("c02", 2), ("c03", 2), ("c27", 2), ("c52", 3)]
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, batch in cases:
            op = convs[name]
            role = roles[name]
            if role["variant"] == "bnprelu":
                w, table = E.fold_bn_conv(role["bn"].scale, role["bn"].shift, op.weight, op.bias)
            elif op.ksize == 1:
                w, table = E.centre_tap(op.weight), op.bias[None, :].astype(np.float64)
            else:
                w, table = op.weight, op.bias[None, :].astype(np.float64)
            w16, info = E.pack(w)
            cp, k_pad, cout_pad = info["cin_pad"], info["k_pad"], info["cout_pad"]
            _, cin, h, wd = graph.shapes[op.inputs[0]]
            cs = E.storage_stride(cin)
            s = op.stride
            ho, wo = h // s, wd // s
            m = batch * ho * wo

            x = (rng.standard_normal((batch, cin, h, wd)) * 0.5).astype(np.float32)
            x_nhwc = np.zeros((batch, h, wd, cs), np.float16)
            x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)

            h_im = tmp / f"im2col_{name}.hsaco"
            compile_kernel(ROOT / "kernels/im2col_f16.loom", "arcface_im2col_f16",
                           {f"{IM}.height": h, f"{IM}.width": wd, f"{IM}.stride": s,
                            f"{IM}.cin_pad": cp, f"{IM}.cin_stride": cs, f"{IM}.k_pad": k_pad}, h_im)
            (cols,), t_im = launch(h_im, "arcface_im2col_f16", (batch * ho, 1, 1), (256, 1, 1),
                                   [("i32", m), ("in_f16", x_nhwc.reshape(-1, cs)),
                                    ("out_f16", ((m, k_pad), np.float16))], tmp, repeat=5)

            # the matmul takes one bias row; the border-class table is added below
            b32 = E.pad_vec(table[table.shape[0] // 2] if table.shape[0] == 9 else table[0], cout_pad)
            h_mm = tmp / f"mm_{name}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_bias_f16_wmma_af16_cf16.loom",
                           "dinov3_matmul_bias_f16_wmma_af16_cf16",
                           {f"{MM}.k_size": k_pad, f"{MM}.n_size": cout_pad}, h_mm)
            (y,), t_mm = launch(h_mm, "dinov3_matmul_bias_f16_wmma_af16_cf16",
                                (cout_pad // 64, (m + 63) // 64, 1), (256, 1, 1),
                                [("i32", m), ("in_f16", cols), ("in_f16", w16), ("in", b32),
                                 ("out_f16", ((m, cout_pad), np.float16))], tmp, repeat=5)
            got = y[:, :op.cout].astype(np.float64)
            x64 = x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2)
            if role["variant"] == "bnprelu":
                # the kernel added the interior row (class 4); swap in the pixel's own class
                cls = np.tile(E.border_class(ho, wo), batch)
                got += table[cls][:, :op.cout] - table[4][None, :op.cout]
                bn = role["bn"]
                want = R.conv2d(R.batchnorm(x64, bn.scale, bn.shift), op.weight, op.bias, 1, 1)
            else:
                want = R.conv2d(x64, op.weight, op.bias, s, op.pad)
            want = want.transpose(0, 2, 3, 1).reshape(m, op.cout)
            flops = 2.0 * m * k_pad * cout_pad
            ok &= report(f"{name} {role['variant']:<7s} {cin:>3d}->{op.cout:<3d} {h}x{wd} s{s} B={batch}  "
                         f"(im2col {t_im['per_launch_us']:7.1f} us + matmul {t_mm['per_launch_us']:7.1f} us, "
                         f"{flops / (t_mm['per_launch_us'] * 1e-6) / 1e12:5.1f} TFLOP/s)",
                         got, want, atol=5e-2, rtol=5e-2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
