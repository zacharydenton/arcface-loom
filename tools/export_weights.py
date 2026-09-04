"""Export w600k_r50's weights in the layout the Loom kernels read, with every
BatchNormalization folded away exactly.

Every conv becomes one f16 matrix W[Cout_pad][K_pad] in the implicit-GEMM gather
order  k = (dy*3 + dx) * Cin_pad + c  (ONNX stores [Cout][Cin][kh][kw]), plus f32
per-channel vectors: the bias and, where the graph has them, the PReLU slope.
Cin is padded to a multiple of 8 so a vector<8xf16> gather never straddles a
tap; K to a multiple of 32 for the k-loop; Cout to a multiple of 64, the conv's
N tile. Padding is zeros, so results are unchanged.

The folds (all in float64, tested to 1e-10 by tools/test_export_fold.py):

  BN -> Conv3x3 (s1, pad 1)   The BN's scale goes into the weights' input channels.
                              Its shift would go into the bias, except that the
                              conv zero-pads *after* the BN, so a border pixel
                              misses the shift of its outside taps. The bias is
                              therefore a [9][Cout] table indexed by the output
                              pixel's border class (top/mid/bottom x left/mid/right),
                              which the `bnprelu` epilogue picks per row. Exact.
  Conv1x1 s2 (shortcut)       Placed in the centre tap of a 3x3 stride-2 layout, which
                              reads (2yo, 2xo), never outside, so the one conv kernel
                              runs every conv.
  BN -> Flatten -> Gemm -> BN The 7x7 BN into the Gemm's columns, the 512-d BN into its
                              rows and bias; columns permuted from ONNX's NCHW flatten
                              (c*49 + p) to the NHWC tensor the kernels hold (p*512 + c).

Output: build/weights/{weights_f16.bin, manifest_f16.txt, weights.bin, manifest.txt,
shapes.txt}, manifest lines `name offset count` in elements, as the siblings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G

ROOT = Path(__file__).resolve().parent.parent
CIN_ALIGN, K_ALIGN, COUT_ALIGN = 8, 32, 64   # the conv gathers vector<8xf16>
BORDER_CLASSES = 9


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def tile_for(n_size: int) -> int:
    """N tile of a 3x3 conv with this padded output width: the 64x128 tile family
    where the width is exactly 128 (scrfd-loom's policy, re-measured here in
    docs/notes.md), 64 otherwise. Either tile pads Cout the same way."""
    return 128 if n_size == 128 else 64


def storage_stride(channels: int) -> int:
    """Physical channel stride of an NHWC activation: every conv writes its
    Cout_pad (64-aligned) columns; the converted stem image is the one 8-wide tensor."""
    return 8 if channels <= 8 else align(channels, COUT_ALIGN)


def border_class(ho: int, wo: int) -> np.ndarray:
    """[ho*wo] int: which of the 9 (dy, dx) tap subsets a stride-1 pad-1 3x3 conv
    sees inside the image at each output pixel: rows 0/1/2 = top/mid/bottom,
    columns 0/1/2 = left/mid/right, class = row*3 + column."""
    assert ho >= 2 and wo >= 2, (ho, wo)
    ry = np.full(ho, 1); ry[0] = 0; ry[-1] = 2
    rx = np.full(wo, 1); rx[0] = 0; rx[-1] = 2
    return (ry[:, None] * 3 + rx[None, :]).reshape(-1)


def taps_inside(cls: int) -> list[int]:
    """The taps dy*3+dx that are inside the image for a border class."""
    ry, rx = divmod(cls, 3)
    dys = [dy for dy in range(3) if not (ry == 0 and dy == 0) and not (ry == 2 and dy == 2)]
    dxs = [dx for dx in range(3) if not (rx == 0 and dx == 0) and not (rx == 2 and dx == 2)]
    return [dy * 3 + dx for dy in dys for dx in dxs]


def fold_bn_conv(scale: np.ndarray, shift: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    """BN(scale, shift) followed by a stride-1 pad-1 3x3 conv (weight [Cout][Cin][3][3],
    bias [Cout]) -> (weight' [Cout][Cin][3][3], bias' [9][Cout]), float64."""
    w = weight.astype(np.float64)
    assert w.shape[2:] == (3, 3), w.shape
    w_scaled = w * scale.reshape(1, -1, 1, 1)
    # per tap, the shift's contribution: [9][Cout]
    per_tap = np.einsum("ocyx,c->oyx", w, shift).reshape(w.shape[0], 9)
    table = np.zeros((BORDER_CLASSES, w.shape[0]), np.float64)
    for cls in range(BORDER_CLASSES):
        table[cls] = bias.astype(np.float64) + per_tap[:, taps_inside(cls)].sum(axis=1)
    return w_scaled, table


def centre_tap(weight: np.ndarray) -> np.ndarray:
    """[Cout][Cin][1][1] -> [Cout][Cin][3][3] with the kernel at (1, 1)."""
    assert weight.shape[2:] == (1, 1), weight.shape
    w = np.zeros(weight.shape[:2] + (3, 3), weight.dtype)
    w[:, :, 1, 1] = weight[:, :, 0, 0]
    return w


def fold_head(bn_in: G.Op, fc: G.Op, bn_out: G.Op, hw: int, channels: int):
    """BN(bn_in) -> Flatten(NCHW) -> Gemm(fc) -> BN(bn_out)  ->  (W [N][K] in NHWC
    column order p*C + c, bias [N]), float64."""
    w = fc.weight.astype(np.float64)                    # [N][K], k = c*hw + p
    n, k = w.shape
    assert k == channels * hw, (k, channels, hw)
    a1 = np.repeat(bn_in.scale, hw)                     # per column k
    s1 = np.repeat(bn_in.shift, hw)
    w1 = w * a1[None, :]
    b1 = fc.bias.astype(np.float64) + w @ s1
    w2 = w1 * bn_out.scale[:, None]
    b2 = bn_out.scale * b1 + bn_out.shift
    # NCHW flatten -> NHWC: column (c*hw + p) moves to (p*C + c)
    w_nhwc = w2.reshape(n, channels, hw).transpose(0, 2, 1).reshape(n, k)
    return w_nhwc, b2


def pack(weight: np.ndarray, cout_align: int | None = None, cin_align: int = CIN_ALIGN) -> tuple[np.ndarray, dict]:
    """[Cout][Cin][3][3] -> W16[Cout_pad][K_pad] in gather order, plus shape info."""
    cout, cin, k, _ = weight.shape
    assert k == 3, weight.shape
    cout_pad = align(cout, cout_align or COUT_ALIGN)
    cin_pad = align(cin, cin_align)
    k_pad = align(k * k * cin_pad, K_ALIGN)
    w = np.zeros((cout_pad, k * k, cin_pad), np.float64)
    w[:cout, :, :cin] = weight.astype(np.float64).transpose(0, 2, 3, 1).reshape(cout, k * k, cin)   # [co][dy*3+dx][c]
    w = w.reshape(cout_pad, k * k * cin_pad)
    w = np.concatenate([w, np.zeros((cout_pad, k_pad - w.shape[1]), np.float64)], axis=1)
    info = dict(cout=cout, cin=cin, k=k, cin_pad=cin_pad, cin_stride=storage_stride(cin),
                k_pad=k_pad, cout_pad=cout_pad)
    return w.astype(np.float16), info


def pad_vec(v: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros(v.shape[:-1] + (width,), np.float32)
    out[..., :v.shape[-1]] = v
    return out


def conv_roles(graph: G.Graph) -> dict[str, dict]:
    """conv name -> {variant, bn (Op|None), prelu (Op|None), add (Op|None)}: what
    the graph attaches to each conv, decided once here and shared with the launch
    table so the export and the schedule cannot disagree."""
    roles = {}
    for op in graph.convs:
        src = graph.producer(op.inputs[0])
        bn = src if src is not None and src.kind == "bn" else None
        cons = graph.consumers(op.output)
        prelu = next((c for c in cons if c.kind == "prelu"), None)
        add = next((c for c in cons if c.kind == "add" and c.inputs[0] == op.output), None)
        if op.ksize == 1:
            variant = "plain"                    # the stride-2 shortcut, centre tap
        elif add is not None:
            variant = "add"
        elif bn is not None:
            variant = "bnprelu"
        else:
            variant = "prelu"
        assert (variant == "plain") == (op.ksize == 1)
        assert (variant in ("prelu", "bnprelu")) == (prelu is not None), op.name
        assert (variant == "bnprelu") == (bn is not None), op.name
        roles[op.name] = dict(variant=variant, bn=bn, prelu=prelu, add=add)
    return roles


def head_ops(graph: G.Graph) -> tuple[G.Op, G.Op, G.Op]:
    """(BN before the flatten, the Gemm, the BN after it)."""
    fc = next(op for op in graph.ops if op.kind == "gemm")
    flat = graph.producer(fc.inputs[0])
    bn_in = graph.producer(flat.inputs[0])
    bn_out = graph.consumers(fc.output)[0]
    assert flat.kind == "flatten" and bn_in.kind == "bn" and bn_out.kind == "bn"
    return bn_in, fc, bn_out


def main() -> None:
    graph = G.load()
    out_dir = ROOT / "build/weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16: list[np.ndarray] = []; f32: list[np.ndarray] = []
    man16: list[str] = []; man32: list[str] = []
    off16 = off32 = 0
    shapes: list[str] = []

    def emit16(name: str, w16: np.ndarray) -> None:
        nonlocal off16
        f16.append(w16.ravel()); man16.append(f"{name} {off16} {w16.size}"); off16 += w16.size

    def emit32(name: str, v: np.ndarray) -> None:
        nonlocal off32
        v = np.ascontiguousarray(v, np.float32)
        f32.append(v.ravel()); man32.append(f"{name} {off32} {v.size}"); off32 += v.size

    roles = conv_roles(graph)
    for op in graph.convs:
        role = roles[op.name]
        if role["variant"] == "bnprelu":
            w, table = fold_bn_conv(role["bn"].scale, role["bn"].shift, op.weight, op.bias)
        elif op.ksize == 1:
            w, table = centre_tap(op.weight), op.bias[None, :].astype(np.float64)
        else:
            w, table = op.weight, op.bias[None, :].astype(np.float64)
        w16, info = pack(w)
        emit16(op.name, w16)
        emit32(f"{op.name}_b", pad_vec(table, info["cout_pad"]))
        if role["prelu"] is not None:
            emit32(f"{op.name}_slope", pad_vec(role["prelu"].slope, info["cout_pad"]))
        info["variant"] = role["variant"]; info["bias_rows"] = table.shape[0]
        info["stride"] = op.stride
        shapes.append(f"{op.name} " + " ".join(f"{k}={v}" for k, v in info.items()))

    bn_in, fc, bn_out = head_ops(graph)
    _, c, h, w_ = graph.shapes[bn_in.output]
    w_fc, b_fc = fold_head(bn_in, fc, bn_out, h * w_, c)
    assert w_fc.shape[1] % K_ALIGN == 0 and w_fc.shape[0] % COUT_ALIGN == 0, w_fc.shape
    emit16("fc", w_fc.astype(np.float16))
    emit32("fc_b", b_fc)
    shapes.append(f"fc k={w_fc.shape[1]} n={w_fc.shape[0]} variant=head bias_rows=1")

    np.concatenate(f16).astype(np.float16).tofile(out_dir / "weights_f16.bin")
    np.concatenate(f32).astype(np.float32).tofile(out_dir / "weights.bin")
    (out_dir / "manifest_f16.txt").write_text("\n".join(man16) + "\n")
    (out_dir / "manifest.txt").write_text("\n".join(man32) + "\n")
    (out_dir / "shapes.txt").write_text("\n".join(shapes) + "\n")
    print(f"{len(man16)} weight matrices, {off16 * 2 / 1e6:.1f} MB f16; {len(man32)} f32 vectors, "
          f"{off32 * 4 / 1e3:.0f} KB -> {out_dir}")


if __name__ == "__main__":
    main()
