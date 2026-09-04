"""The w600k_r50 graph, read from the ONNX file and resolved at its fixed input size.

This is the single source of truth: the float64 reference interprets it, the
weight export walks it, and the host's launch table is generated from it, so
the 53 convolutions cannot drift between the three.

The graph is a plain iResNet-50: a stem conv, 24 residual blocks of
`BN -> Conv3x3 -> PRelu -> Conv3x3 -> Add`, and a head of
`BN -> Flatten -> Gemm -> BN`. Every BatchNormalization is inference-mode and
is stored here as the per-channel affine `scale * x + shift` it amounts to,
in float64, so the export can fold it exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
INPUT_SIZE = 112
EMBEDDING = 512


def model_path() -> Path:
    return Path(os.environ.get("ARCFACE_ONNX", DEFAULT_MODEL)).expanduser()


def require(condition: bool, detail) -> None:
    """Reject an incompatible graph even when Python assertions are disabled."""
    if not condition:
        raise ValueError(f"unsupported ArcFace ONNX graph: {detail}")


@dataclass
class Op:
    kind: str                 # conv, bn, prelu, add, flatten, gemm
    name: str                 # conv: c00..c52; gemm: fc; bn: bn00..; others: the ONNX output name
    inputs: list[str]
    output: str
    # conv and gemm
    weight: np.ndarray | None = None      # conv [Cout][Cin][k][k] f32; gemm [N][K] f32 (transB applied)
    bias: np.ndarray | None = None        # [Cout] / [N] f32
    stride: int = 1
    pad: int = 0
    # bn: y = scale * x + shift, float64
    scale: np.ndarray | None = None
    shift: np.ndarray | None = None
    # prelu: y = max(x, 0) + slope * min(x, 0), per channel, float64
    slope: np.ndarray | None = None
    # shapes resolved at the fixed input size, NCHW (or [N, F] after the flatten)
    out_shape: tuple[int, ...] = field(default=())

    @property
    def cout(self) -> int: return int(self.weight.shape[0])
    @property
    def cin(self) -> int: return int(self.weight.shape[1])
    @property
    def ksize(self) -> int: return int(self.weight.shape[2])


@dataclass
class Graph:
    ops: list[Op]
    input: str
    output: str                        # the one ONNX output: [N, 512]
    shapes: dict[str, tuple[int, ...]]  # every tensor at the fixed size
    size: int

    @property
    def convs(self) -> list[Op]:
        return [op for op in self.ops if op.kind == "conv"]

    def by_output(self, name: str) -> Op:
        return next(op for op in self.ops if op.output == name)

    def consumers(self, name: str) -> list[Op]:
        return [op for op in self.ops if name in op.inputs]

    def producer(self, name: str) -> Op | None:
        return next((op for op in self.ops if op.output == name), None)


def load(size: int = INPUT_SIZE, path: Path | None = None) -> Graph:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path or model_path()))
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    attr = lambda n: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    shapes: dict[str, tuple[int, ...]] = {g.input[0].name: (1, 3, size, size)}
    ops: list[Op] = []
    counts = {"conv": 0, "bn": 0}

    for n in g.node:
        t = n.op_type
        a = attr(n)
        ins = [x for x in n.input if x not in init]
        out = n.output[0]
        src = shapes[ins[0]]

        if t == "Conv":
            w, b = init[n.input[1]], init[n.input[2]]
            s, p, k = a.get("strides", [1])[0], a.get("pads", [0])[0], w.shape[2]
            require(a.get("group", 1) == 1 and a.get("dilations", [1, 1]) == [1, 1],
                    (n.name, "group/dilations", a.get("group", 1), a.get("dilations", [1, 1])))
            require(w.ndim == 4 and w.shape[2] == w.shape[3], (n.name, "weight shape", w.shape))
            require(a.get("strides", [1, 1]) == [s, s], (n.name, "asymmetric strides", a.get("strides")))
            require(a.get("pads", [0, 0, 0, 0]) == [p, p, p, p], (n.name, "asymmetric pads", a.get("pads")))
            require((k, p) in ((3, 1), (1, 0)), (n.name, k, p))
            N, C, H, W = src
            require(C == w.shape[1], (n.name, C, w.shape))
            ho, wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
            shapes[out] = (N, int(w.shape[0]), ho, wo)
            ops.append(Op("conv", f"c{counts['conv']:02d}", ins, out, weight=w.astype(np.float32),
                          bias=b.astype(np.float32), stride=s, pad=p, out_shape=shapes[out]))
            counts["conv"] += 1
        elif t == "BatchNormalization":
            gamma, beta, mean, var = (init[x].astype(np.float64) for x in n.input[1:5])
            eps = float(a.get("epsilon", 1e-5))
            scale = gamma / np.sqrt(var + eps)
            shift = beta - mean * scale
            shapes[out] = src
            ops.append(Op("bn", f"bn{counts['bn']:02d}", ins, out, scale=scale, shift=shift, out_shape=src))
            counts["bn"] += 1
        elif t == "PRelu":
            slope = init[n.input[1]].astype(np.float64).reshape(-1)
            require(slope.size == src[1], (n.name, slope.shape, src))
            shapes[out] = src
            ops.append(Op("prelu", out, ins, out, slope=slope, out_shape=src))
        elif t == "Add":
            require(shapes[ins[0]] == shapes[ins[1]], (out, shapes[ins[0]], shapes[ins[1]]))
            shapes[out] = src
            ops.append(Op("add", out, ins, out, out_shape=src))
        elif t == "Flatten":
            require(a.get("axis", 1) == 1, a)
            N, C, H, W = src
            shapes[out] = (N, C * H * W)
            ops.append(Op("flatten", out, ins, out, out_shape=shapes[out]))
        elif t == "Gemm":
            require(a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0, a)
            require(a.get("transA", 0) == 0 and a.get("transB", 0) == 1, a)
            w, b = init[n.input[1]], init[n.input[2]]          # [N][K], [N]
            require(src[1] == w.shape[1], (src, w.shape))
            shapes[out] = (src[0], int(w.shape[0]))
            ops.append(Op("gemm", "fc", ins, out, weight=w.astype(np.float32), bias=b.astype(np.float32),
                          out_shape=shapes[out]))
        else:
            raise NotImplementedError(t)

    require(len(g.output) == 1, [o.name for o in g.output])
    graph = Graph(ops=ops, input=g.input[0].name, output=g.output[0].name, shapes=shapes, size=size)
    _check_structure(graph)
    return graph


def _check_structure(graph: Graph) -> None:
    """The folds in export_weights.py and gen_launch_table.py rely on these."""
    for op in graph.ops:
        cons = graph.consumers(op.output)
        if op.kind == "bn":
            # A BN is folded into its one consumer: a stride-1 3x3 conv (border-bias
            # fold), the Flatten (into the Gemm), or nothing (the final BN, into the
            # Gemm's rows).
            require(len(cons) <= 1, (op.name, [c.kind for c in cons]))
            if cons:
                c = cons[0]
                require(c.kind == "flatten" or (c.kind == "conv" and c.ksize == 3 and c.stride == 1),
                        (op.name, c.kind))
            else:
                require(op.output == graph.output, op.name)
        elif op.kind == "prelu":
            # PRelu folds into the epilogue of the conv that produces its input.
            p = graph.producer(op.inputs[0])
            require(p is not None and p.kind == "conv" and len(graph.consumers(p.output)) == 1, op.name)
        elif op.kind == "add":
            # Hosted on the conv producing its *first* input; the second is the shortcut.
            p = graph.producer(op.inputs[0])
            require(p is not None and p.kind == "conv" and p.ksize == 3 and len(graph.consumers(p.output)) == 1,
                    op.name)
        elif op.kind == "conv" and op.ksize == 1:
            require(op.stride == 2, op.name)


if __name__ == "__main__":
    graph = load()
    print(f"{len(graph.ops)} ops, {len(graph.convs)} convs at {graph.size}x{graph.size}")
    for op in graph.ops:
        detail = ""
        if op.kind == "conv":
            detail = f"{op.cin:>3d}->{op.cout:<3d} k{op.ksize} s{op.stride}"
        elif op.kind == "gemm":
            detail = f"{op.weight.shape[1]}->{op.weight.shape[0]}"
        elif op.kind == "bn":
            detail = f"scale[{op.scale.min():.2e}, {op.scale.max():.2e}]"
        print(f"  {op.name:>6s} {op.kind:<8s} {str(op.inputs):<22s} -> {op.output:<5s} {str(op.out_shape):<22s} {detail}")
