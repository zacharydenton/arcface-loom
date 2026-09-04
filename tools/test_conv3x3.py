"""conv3x3_f16_wmma (implicit GEMM) and its epilogue variants vs the float64
reference, on tiny planes that hit all nine border classes and on real layers
with the real folded weights and real activations.

The synthetic cases grade the kernel against the exported operands (weights,
bias table, slope, residual); the real `bnprelu` layers are graded against the
unfolded graph -- BN, conv, PReLU -- so the fold and the kernel are checked
together on what the model actually feeds them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
import export_weights as E
import test_align as TA


def run_case(tmp, name, x_nhwc, cin, weight, table, stride, variant="plain", tile=64,
             slope=None, residual=None, want=None, repeat=10):
    """x_nhwc [B,H,W,cs] f16; weight [cout][cin][3][3]; table [rows][cout] (rows 1 or 9)."""
    cout = weight.shape[0]
    w16, info = E.pack(weight, cout_align=tile)
    cp, k_pad, cout_pad = info["cin_pad"], info["k_pad"], info["cout_pad"]
    batch, h, w, cs = x_nhwc.shape
    ho, wo = h // stride, w // stride
    m = batch * ho * wo
    b32 = E.pad_vec(table, cout_pad)

    suffix = "" if variant == "plain" else f"_{variant}"
    base = "conv3x3_n128_f16_wmma" if tile == 128 else "conv3x3_f16_wmma"
    ns, symbol = f"arcface.{base}{suffix}", f"arcface_{base}{suffix}"
    hsaco = tmp / f"{base}{suffix}_{name.replace(' ', '_')}_{h}x{w}.hsaco"
    compile_kernel(ROOT / f"kernels/{base}{suffix}.loom", symbol,
                   {f"{ns}.height": h, f"{ns}.width": w, f"{ns}.stride": stride,
                    f"{ns}.cin_pad": cp, f"{ns}.cin_stride": cs,
                    f"{ns}.k_size": k_pad, f"{ns}.n_size": cout_pad}, hsaco)
    args = [("i32", m), ("in_f16", x_nhwc.reshape(-1, cs)), ("in_f16", w16), ("in", b32),
            ("out_f16", ((m, cout_pad), np.float16))]
    if variant == "add":
        args.append(("in_f16", residual))
    if variant in ("prelu", "bnprelu"):
        args.append(("in", E.pad_vec(slope, cout_pad)))
    (y,), t = launch(hsaco, symbol, (cout_pad // tile, (m + 63) // 64, 1), (256, 1, 1), args, tmp, repeat=repeat)

    if want is None:
        x64 = x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2)
        want = R.conv2d(x64, weight, np.zeros(cout), stride, 1).transpose(0, 2, 3, 1).reshape(m, cout)
        cls = np.tile(E.border_class(ho, wo), batch) if table.shape[0] == 9 else np.zeros(m, int)
        want = want + table[cls]
        if residual is not None:
            want = want + residual[:, :cout].astype(np.float64)
        if slope is not None:
            want = np.maximum(want, 0) + slope * np.minimum(want, 0)
    flops = 2.0 * m * 9 * cin * cout
    us = t["per_launch_us"]
    return report(f"{name:<7s} {'n128 ' if tile == 128 else ''}{variant:<7s} {cin:>3d}->{cout:<3d} {h:>3d}x{w:<3d} s{stride} B={batch}  "
                  f"({us:8.1f} us, {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s useful)",
                  y[:, :cout], want, atol=5e-2, rtol=5e-2)


def synthetic(rng, batch, cin, cout, h, w, cs=None):
    cs = cs or E.storage_stride(cin)
    x = np.zeros((batch, h, w, cs), np.float16)
    x[..., :cin] = (rng.standard_normal((batch, h, w, cin)) * 0.5).astype(np.float16)
    weight = (rng.standard_normal((cout, cin, 3, 3)) * (1.0 / np.sqrt(9 * cin))).astype(np.float32)
    return x, weight


def main() -> int:
    ok = True
    rng = np.random.default_rng(21)
    graph = G.load()
    convs = {op.name: op for op in graph.convs}
    roles = E.conv_roles(graph)
    with workdir() as tmp:
        tmp = Path(tmp)
        # Tiny planes: 4x6 and 3x5 hit all nine border classes; stride 2 on 4x6.
        for variant in ("plain", "prelu", "bnprelu", "add"):
            for (h, w, stride, cin, cout, tile) in ((4, 6, 1, 8, 64, 64), (3, 5, 1, 8, 64, 64), (4, 6, 2, 8, 64, 64),
                                                    (4, 6, 1, 8, 128, 128), (3, 5, 1, 16, 128, 128)):
                if variant == "bnprelu" and stride == 2:
                    continue
                x, weight = synthetic(rng, 2, cin, cout, h, w)
                rows = 9 if variant == "bnprelu" else 1
                table = rng.standard_normal((rows, cout)) * 0.1
                slope = rng.uniform(0.05, 0.5, cout) if variant in ("prelu", "bnprelu") else None
                m = 2 * (h // stride) * (w // stride)
                residual = rng.standard_normal((m, cout)).astype(np.float16) if variant == "add" else None
                ok &= run_case(tmp, "tiny", x, cin, weight, table, stride, variant, tile, slope, residual)

        # Real layers on real activations: one face through the reference.
        import cv2
        crops = TA.crops(cv2.imread(str(TA.test_image())), TA.fixture())[:1]
        tensors = R.forward(graph, R.blob_from_bgr_f64(crops), keep=True)
        for name, batch in (("c00", 2), ("c01", 2), ("c02", 2), ("c03", 2), ("c11", 2), ("c12", 2),
                            ("c27", 3), ("c49", 3), ("c52", 3)):
            op, role = convs[name], roles[name]
            src = op.inputs[0] if role["bn"] is None else role["bn"].inputs[0]
            x0 = tensors[src]                                          # [1,cin,h,w] f64, pre-BN
            _, cin, h, w = x0.shape
            x = np.concatenate([x0 * s for s in (1.0, 0.7, -0.4)[:batch]])
            cs = E.storage_stride(cin)
            x_nhwc = np.zeros((batch, h, w, cs), np.float16)
            x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)
            x64 = x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2)
            variant = role["variant"]
            tile = E.tile_for(E.align(op.cout, E.COUT_ALIGN))
            slope = role["prelu"].slope if role["prelu"] is not None else None
            residual = None
            if variant == "bnprelu":
                weight, table = E.fold_bn_conv(role["bn"].scale, role["bn"].shift, op.weight, op.bias)
                want = R.prelu(R.conv2d(R.batchnorm(x64, role["bn"].scale, role["bn"].shift), op.weight, op.bias, 1, 1), slope)
            elif op.ksize == 1:
                weight, table = E.centre_tap(op.weight), op.bias[None, :].astype(np.float64)
                want = R.conv2d(x64, op.weight, op.bias, 2, 0)
            else:
                weight, table = op.weight, op.bias[None, :].astype(np.float64)
                want = R.conv2d(x64, op.weight, op.bias, op.stride, 1)
                if variant == "prelu":
                    want = R.prelu(want, slope)
            ho, wo = h // op.stride, w // op.stride
            want = want.transpose(0, 2, 3, 1).reshape(batch * ho * wo, op.cout)
            if variant == "add":
                residual = rng.standard_normal((batch * ho * wo, E.align(op.cout, E.COUT_ALIGN))).astype(np.float16)
                want = want + residual[:, :op.cout].astype(np.float64)
            ok &= run_case(tmp, name, x_nhwc, cin, weight, table, op.stride, variant, tile, slope, residual, want)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
