"""Turn the graph into the host's launch schedule, so the 53 convolutions are
never transcribed by hand.

From `graph.py`'s op list and `export_weights.py`'s roles this produces:

  host/graph_table.inc      the ordered launch table as C struct literals
  build/launch_table.json   the same, for the Python tests
  scripts/kernels.generated the build lines for every distinct (kernel, config)

What the schedule folds away:

  BatchNormalization -> its one consumer conv's weights and [9][n] border-bias
              table (export), run as the `bnprelu` variant; the head's two BNs
              into the Gemm.
  PRelu    -> the `prelu` / `bnprelu` variant of the conv that produces its input.
  Add      -> hosted on the conv producing its *first* input as the `add` variant,
              with the shortcut tensor as `residual`. That conv is scheduled after
              the shortcut exists, so a downsample block's 1x1 runs before the 3x3
              that hosts the add.
  Conv1x1 s2 -> a centre-tap 3x3 stride-2 conv (export), the `plain` variant.
  Flatten + Gemm -> the split-K matmul on the last residual tensor, whose NHWC
              rows are already the flattened [B][25088], then the f32 reduce.

Every launch carries an ordered list of the extra buffers its kernel takes after
(m, a, w, bias, c): `residual` then `slope`, and the host appends them in that
order, so the kernarg layout is generated too.

Buffers are assigned by liveness: a tensor's buffer is free after its last
consumer, and freed buffers are reused by size.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
from export_weights import (align, storage_stride, tile_for, conv_roles, head_ops, head_splits,
                            CIN_ALIGN, K_ALIGN, COUT_ALIGN, BORDER_CLASSES)

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"convert": 0, "conv3x3": 1, "head_matmul": 2, "head_reduce": 3}
VARIANTS = {"": 0, "plain": 0, "prelu": 1, "bnprelu": 2, "add": 3}


@dataclass
class Launch:
    kind: str                 # convert, conv3x3, head_matmul, head_reduce
    variant: str              # conv3x3: plain|prelu|bnprelu|add
    name: str                 # weight name (convs, head) or a label
    src: str                  # input tensor
    dst: str                  # output tensor
    extra: str = ""           # residual tensor (add)
    stage: str = ""           # profiler label
    # geometry of the *input* for convs, and of the output plane
    h: int = 0
    w: int = 0
    stride: int = 1
    cin: int = 0
    cin_pad: int = 0
    cin_stride: int = 0
    k_size: int = 0
    cout: int = 0
    n_size: int = 0
    ho: int = 0
    wo: int = 0
    tile: int = 64            # N tile of a 3x3 conv: tile_for(n_size)
    splits: int = 1           # head only
    bias_rows: int = 1        # 1, or 9 for the border-class table
    aux: list[str] = field(default_factory=list)   # extra kernargs after c: residual, slope
    src_buf: int = -1
    dst_buf: int = -1
    extra_buf: int = -1

    @property
    def stem(self) -> str:
        """The HSACO this launch runs: one per distinct kernel+config."""
        if self.kind == "convert":
            return "hwc_u8_to_nhwc"
        if self.kind == "head_matmul":
            return f"head_splitk_k{self.k_size}_n{self.n_size}_s{self.splits}"
        if self.kind == "head_reduce":
            return f"head_reduce_n{self.n_size}_s{self.splits}"
        wide = "n128_" if self.tile == 128 else ""
        return (f"conv3x3_{wide}{self.variant}_{self.h}x{self.w}_s{self.stride}"
                f"_c{self.cin_pad}of{self.cin_stride}_k{self.k_size}_n{self.n_size}")

    @property
    def symbol(self) -> str:
        if self.kind == "convert":
            return "arcface_hwc_u8_to_nhwc_f16"
        if self.kind == "head_matmul":
            return "arcface_matmul_splitk_f16_wmma"
        if self.kind == "head_reduce":
            return "arcface_splitk_reduce_f32"
        base = "conv3x3_n128_f16_wmma" if self.tile == 128 else "conv3x3_f16_wmma"
        suffix = "" if self.variant == "plain" else f"_{self.variant}"
        return f"arcface_{base}{suffix}"


def build_schedule(graph: G.Graph):
    """Returns (ordered launches, tensor -> (rows per image, columns, element bytes), buffer bytes per image)."""
    roles = conv_roles(graph)
    producer = {op.output: op for op in graph.ops}

    # alias: tensor name -> the tensor whose buffer actually holds it (after folding)
    alias: dict[str, str] = {}
    def resolve(t: str) -> str:
        while t in alias:
            t = alias[t]
        return t

    launches: list[Launch] = []
    shapes: dict[str, tuple[int, int, int]] = {}   # tensor -> (rows per image, cols, elem bytes)

    _, cin0, H, W = graph.shapes[graph.input]
    conv_in = "nhwc_input"
    shapes[conv_in] = (H * W, storage_stride(cin0), 2)
    launches.append(Launch("convert", "", "convert", graph.input, conv_in, stage="convert",
                           h=H, w=W, cin=cin0, cin_pad=storage_stride(cin0), cin_stride=storage_stride(cin0), ho=H, wo=W))
    alias[graph.input] = conv_in

    for op in graph.ops:
        if op.kind == "conv":
            role = roles[op.name]
            variant, extra, out, aux = role["variant"], "", op.output, []
            # The BN feeding this conv is folded: the conv reads the BN's input.
            src = role["bn"].inputs[0] if role["bn"] is not None else op.inputs[0]
            if role["add"] is not None:
                extra = role["add"].inputs[1]
                out = role["add"].output
                alias[out] = op.output
                aux.append("residual")
            if role["prelu"] is not None:
                alias[role["prelu"].output] = op.output
                aux.append("slope")
            _, cin, h, w = graph.shapes[src]
            cs = shapes[resolve(src)][1]
            assert cs == storage_stride(cin), (op.name, cs, storage_stride(cin))
            if op.stride == 2:
                assert h % 2 == 0 and w % 2 == 0, (op.name, h, w)
            cin_pad = align(cin, CIN_ALIGN)
            k_size = align(9 * cin_pad, K_ALIGN)
            n_size = align(op.cout, COUT_ALIGN)
            ho, wo = op.out_shape[2], op.out_shape[3]
            launches.append(Launch("conv3x3", variant, op.name, src, op.output, extra,
                                   stage=f"conv3x3 {h}x{w} {variant}",
                                   h=h, w=w, stride=op.stride, cin=cin, cin_pad=cin_pad, cin_stride=cs,
                                   k_size=k_size, cout=op.cout, n_size=n_size, ho=ho, wo=wo,
                                   tile=tile_for(n_size), bias_rows=BORDER_CLASSES if variant == "bnprelu" else 1,
                                   aux=aux))
            shapes[op.output] = (ho * wo, n_size, 2)
        elif op.kind in ("bn", "prelu", "add", "flatten"):
            continue                                 # folded (asserted by conv_roles / graph)
        elif op.kind == "gemm":
            bn_in, fc, bn_out = head_ops(graph)
            src = bn_in.inputs[0]                    # the last residual tensor, NHWC [B*49][512]
            rows, cols, _ = shapes[resolve(src)]
            k_size, n_size = rows * cols, align(fc.cout, COUT_ALIGN)
            assert k_size == fc.weight.shape[1] and k_size % 128 == 0, (k_size, fc.weight.shape)
            splits = head_splits(k_size)
            launches.append(Launch("head_matmul", "", "fc", src, "partials", stage="head matmul",
                                   k_size=k_size, cout=fc.cout, n_size=n_size, ho=1, wo=1, splits=splits))
            shapes["partials"] = (splits, n_size, 4)
            launches.append(Launch("head_reduce", "", "fc", "partials", "embedding", stage="head reduce",
                                   n_size=n_size, ho=1, wo=1, splits=splits))
            shapes["embedding"] = (1, n_size, 4)
            alias[bn_out.output] = "embedding"
        else:
            raise NotImplementedError(op.kind)

    # --- schedule: a launch runs once every tensor it reads exists --------------------
    for l in launches:
        if l.kind == "convert":
            continue
        l.src = resolve(l.src)
        l.extra = resolve(l.extra) if l.extra else ""
    ordered: list[Launch] = [launches[0]]
    ready = {conv_in, graph.input}
    pending = list(launches[1:])
    while pending:
        progressed = False
        for l in list(pending):
            needs = {l.src} | ({l.extra} if l.extra else set())
            if needs <= ready:
                ordered.append(l); pending.remove(l); ready.add(l.dst); progressed = True
                break                      # keep graph order as the tie-break
        if not progressed:
            raise RuntimeError("schedule stuck: " + ", ".join(f"{l.name}<-{l.src}/{l.extra}" for l in pending))

    # --- buffers by liveness ------------------------------------------------------------
    last_use: dict[str, int] = {}
    for i, l in enumerate(ordered):
        for t in (l.src, l.extra):
            if t:
                last_use[t] = i
    last_use["embedding"] = len(ordered)             # the output lives to the end
    def bytes_of(t: str) -> int:
        rows, cols, elem = shapes[t]
        return rows * cols * elem
    buffers: list[int] = []            # size in bytes per image
    free: list[int] = []
    owner: dict[str, int] = {}
    for i, l in enumerate(ordered):
        need = bytes_of(l.dst)
        pick = None
        for b in sorted(free, key=lambda b: buffers[b]):
            if buffers[b] >= need:
                pick = b; break
        if pick is None:
            pick = len(buffers); buffers.append(need)
        else:
            free.remove(pick)
        owner[l.dst] = pick
        l.dst_buf = pick
        l.src_buf = owner[l.src] if l.src in owner else -1     # -1: the raw input
        l.extra_buf = owner[l.extra] if l.extra else -1
        assert l.dst_buf != l.src_buf and l.dst_buf != l.extra_buf, ("in-place hazard", l.name)
        for t in (l.src, l.extra):
            if t and t in owner and last_use.get(t) == i and t != "embedding":
                free.append(owner[t])
    return ordered, shapes, buffers


def emit(ordered: list[Launch], shapes, buffers) -> None:
    stems = sorted({l.stem for l in ordered})
    out = next(l for l in ordered if l.dst == "embedding")
    lines = ["// GENERATED by tools/gen_launch_table.py from the ONNX graph. Do not edit.",
             f"#define ARCFACE_LAUNCH_COUNT {len(ordered)}",
             f"#define ARCFACE_BUFFER_COUNT {len(buffers)}",
             f"#define ARCFACE_KERNEL_COUNT {len(stems)}",
             f"#define ARCFACE_OUTPUT_BUFFER {out.dst_buf}",
             f"#define ARCFACE_EMBEDDING {out.n_size}",
             "// bytes per image, per buffer",
             "static const size_t arcface_buffer_bytes[ARCFACE_BUFFER_COUNT] = {" + ", ".join(str(b) for b in buffers) + "};",
             "static const char *const arcface_kernel_stems[ARCFACE_KERNEL_COUNT] = {" + ", ".join(f'"{s}"' for s in stems) + "};",
             "static const char *const arcface_kernel_symbols[ARCFACE_KERNEL_COUNT] = {"
             + ", ".join(f'"{next(l.symbol for l in ordered if l.stem == s)}"' for s in stems) + "};",
             "// kind: 0 convert, 1 conv3x3, 2 head matmul (split-K), 3 head reduce; "
             "variant: 0 plain, 1 prelu, 2 bnprelu, 3 add; aux: bit 0 residual, bit 1 slope, in that order",
             "static const arcface_launch arcface_launches[ARCFACE_LAUNCH_COUNT] = {"]
    for l in ordered:
        aux = (1 if "residual" in l.aux else 0) | (2 if "slope" in l.aux else 0)
        lines.append(f'  {{{KINDS[l.kind]}, {VARIANTS[l.variant]}, {stems.index(l.stem)}, "{l.name}", "{l.stage}", '
                     f'{l.h}, {l.w}, {l.stride}, {l.cin_pad}, {l.cin_stride}, {l.k_size}, {l.n_size}, {l.ho}, {l.wo}, '
                     f'{l.tile}, {l.splits}, {l.bias_rows}, {aux}, {l.src_buf}, {l.dst_buf}, {l.extra_buf}}},')
    lines.append("};")
    (ROOT / "host/graph_table.inc").write_text("\n".join(lines) + "\n")

    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build/launch_table.json").write_text(json.dumps(
        {"launches": [l.__dict__ | {"stem": l.stem, "symbol": l.symbol} for l in ordered],
         "buffers": buffers, "shapes": shapes}, indent=1))

    build = ["# GENERATED by tools/gen_launch_table.py. Sourced by scripts/build_kernels.sh."]
    seen = set()
    for l in ordered:
        if l.stem in seen:
            continue
        seen.add(l.stem)
        if l.kind == "convert":
            continue   # hand-listed in build_kernels.sh
        if l.kind == "head_matmul":
            ns = "arcface.matmul_splitk_f16_wmma"
            build.append(f"compile matmul_splitk_f16_wmma {l.symbol} {l.stem} {ns}.k_size={l.k_size} "
                         f"{ns}.n_size={l.n_size} {ns}.splits={l.splits}")
        elif l.kind == "head_reduce":
            ns = "arcface.splitk_reduce_f32"
            build.append(f"compile splitk_reduce_f32 {l.symbol} {l.stem} {ns}.n_size={l.n_size} {ns}.splits={l.splits}")
        else:
            src = l.symbol[len("arcface_"):]
            ns = "arcface." + src
            build.append(f"compile {src} {l.symbol} {l.stem} {ns}.height={l.h} {ns}.width={l.w} {ns}.stride={l.stride} "
                         f"{ns}.cin_pad={l.cin_pad} {ns}.cin_stride={l.cin_stride} {ns}.k_size={l.k_size} {ns}.n_size={l.n_size}")
    (ROOT / "scripts/kernels.generated").write_text("\n".join(build) + "\n")


def main() -> None:
    graph = G.load()
    ordered, shapes, buffers = build_schedule(graph)
    emit(ordered, shapes, buffers)
    per_image = sum(buffers)
    stems = {l.stem for l in ordered}
    by_kind = {}
    for l in ordered:
        by_kind[f"{l.kind}/{l.variant}"] = by_kind.get(f"{l.kind}/{l.variant}", 0) + 1
    print(f"{len(ordered)} launches, {len(stems)} distinct kernels, {len(buffers)} buffers, "
          f"{per_image / 1e6:.2f} MB/image")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:<22s} {v}")


if __name__ == "__main__":
    main()
