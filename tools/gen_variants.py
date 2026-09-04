"""Generate the epilogue variants of the 3x3 conv from the generated bases.

The graph needs three epilogues on the 3x3 conv, plus the plain one for the
centre-tap shortcuts:

  plain     bias only                                the 4 stride-2 shortcuts
  prelu     prelu(x + bias)                          the stem
  bnprelu   prelu(x + bias[class(row)])              the first conv of every block: the
                                                     preceding BN is folded into the
                                                     weights and a [9][n] border-class
                                                     bias table (tools/export_weights.py)
  add       x + bias + residual                      the second conv of every block

`prelu(v) = max(v, 0) + slope[n] * min(v, 0)` with a per-channel f32 slope, computed
as `max(v,0) + slope * (v - max(v,0))` so it needs no min. `residual` is an NHWC f16
tensor of the output's shape. Extra launch operands are appended in the order
(residual, slope), which host/arcface.cpp follows from the generated table.

Each variant is produced by literal anchored edits, the way the siblings' generators
work: if the source moves, the assert fails loudly rather than generating something
subtly wrong.
"""
from __future__ import annotations

import argparse
from pathlib import Path

KERNELS = Path(__file__).resolve().parent.parent / "kernels"
# (source file, export symbol, config namespace)
BASES = (("conv3x3_f16_wmma.loom", "arcface_conv3x3_f16_wmma", "arcface.conv3x3_f16_wmma"),
         ("conv3x3_n128_f16_wmma.loom", "arcface_conv3x3_n128_f16_wmma", "arcface.conv3x3_n128_f16_wmma"))

LAUNCH = "} launch(%m_size: index, %a: buffer, %w: buffer, %bias: buffer, %c: buffer) {"
LAUNCH_RESIDUAL = "} launch(%m_size: index, %a: buffer, %w: buffer, %bias: buffer, %c: buffer, %residual: buffer) {"
LAUNCH_SLOPE = "} launch(%m_size: index, %a: buffer, %w: buffer, %bias: buffer, %c: buffer, %slope: buffer) {"

ZERO4 = "%zero_f16x4 = vector.constant 0.0 : vector<4xf16>"
C_VIEW = "  %c_view = buffer.view %c_global[%c0_offset] : buffer -> view<[%m_size]x[%n_size]xf16>"
BIAS_VIEW = "  %bias_view = buffer.view %bias_global[%c0_offset] : buffer -> view<[%n_size]xf32>"
BIAS_TABLE_VIEW = "  %bias_view = buffer.view %bias_global[%c0_offset] : buffer -> view<[%c9]x[%n_size]xf32>"
RESIDUAL_VIEW = """  %residual_global = buffer.assume.memory_space<global> %residual : buffer
  %residual_view = buffer.view %residual_global[%c0_offset] : buffer -> view<[%m_size]x[%n_size]xf16>"""
SLOPE_VIEW = """  %slope_global = buffer.assume.memory_space<global> %slope : buffer
  %slope_view = buffer.view %slope_global[%c0_offset] : buffer -> view<[%n_size]xf32>"""

BIAS_LOAD = "    %bias_values = vector.load %bias_view[%out_col] : view<[%n_size]xf32> -> vector<4xf32>\n"
BIAS_ADD = "      %biased = vector.addf %values, %bias_values : vector<4xf32>\n"

STORE = """        %narrow = vector.fptrunc %biased : vector<4xf32> to vector<4xf16>
        vector.store %narrow, %c_view[%bounded, %out_col] : vector<4xf16>, view<[%m_size]x[%n_size]xf16>"""

ADD = """        %prior_half = vector.load %residual_view[%bounded, %out_col] : view<[%m_size]x[%n_size]xf16> -> vector<4xf16>
        %prior = vector.extf %prior_half : vector<4xf16> to vector<4xf32>
        %summed = vector.addf %biased, %prior : vector<4xf32>
"""
PRELU = """        %slope_values = vector.load %slope_view[%out_col] : view<[%n_size]xf32> -> vector<4xf32>
        %positive = vector.maxnumf %{src}, %zero_f32x4 : vector<4xf32>
        %negative = vector.subf %{src}, %positive : vector<4xf32>
        %leaked = vector.mulf %negative, %slope_values : vector<4xf32>
        %activated = vector.addf %positive, %leaked : vector<4xf32>
"""
# The border class of this output row: rows 0/1/2 = top/mid/bottom, columns 0/1/2 =
# left/mid/right, class = row*3 + column -- the order tools/export_weights.py
# builds the table in. Then the bias add the base did before the row was known.
CLASS_BIAS = """        %pixel = index.rem %bounded, %plane : index
        %yo = index.div %pixel, %wo : index
        %xo = index.rem %pixel, %wo : index
        %ho_last = index.sub %ho, %c1 : index
        %wo_last = index.sub %wo, %c1 : index
        %y_first = index.cmp eq, %yo, %c0 : index
        %y_last = index.cmp eq, %yo, %ho_last : index
        %x_first = index.cmp eq, %xo, %c0 : index
        %x_last = index.cmp eq, %xo, %wo_last : index
        %ry_rest = scf.select %y_last, %c2, %c1 : index
        %ry = scf.select %y_first, %c0, %ry_rest : index
        %rx_rest = scf.select %x_last, %c2, %c1 : index
        %rx = scf.select %x_first, %c0, %rx_rest : index
        %ry3 = index.mul %ry, %c3 : index
        %class0 = index.add %ry3, %rx : index
        %class = index.assume %class0 [range(%class0, 0, 8)] : index
        %bias_values = vector.load %bias_view[%class, %out_col] : view<[%c9]x[%n_size]xf32> -> vector<4xf32>
        %biased = vector.addf %values, %bias_values : vector<4xf32>
"""

VARIANTS = {
    "prelu": dict(residual=False, slope=True, table=False,
                  body=PRELU.format(src="biased") + STORE.replace("%biased", "%activated")),
    "bnprelu": dict(residual=False, slope=True, table=True,
                    body=CLASS_BIAS + PRELU.format(src="biased") + STORE.replace("%biased", "%activated")),
    "add": dict(residual=True, slope=False, table=False,
                body=ADD + STORE.replace("%biased", "%summed")),
}


def generate(variant: str, text: str, source: str, symbol: str, namespace: str) -> str:
    spec = VARIANTS[variant]
    for anchor in (LAUNCH, C_VIEW, BIAS_VIEW, BIAS_LOAD, BIAS_ADD, STORE, ZERO4):
        assert text.count(anchor) == 1, f"{source}: anchor x{text.count(anchor)}\n{anchor[:90]}"
    out = text
    launch = LAUNCH
    views = C_VIEW
    if spec["residual"]:
        launch = LAUNCH_RESIDUAL
        views += "\n" + RESIDUAL_VIEW
    if spec["slope"]:
        launch = launch.replace(") {", ", %slope: buffer) {")
        views += "\n" + SLOPE_VIEW
    out = out.replace(LAUNCH, launch).replace(C_VIEW, views)
    out = out.replace(ZERO4, ZERO4 + "\n  %zero_f32x4 = vector.constant 0.0 : vector<4xf32>")
    if spec["table"]:
        out = out.replace(BIAS_VIEW, BIAS_TABLE_VIEW).replace(BIAS_LOAD, "").replace(BIAS_ADD, "")
    out = out.replace(STORE, spec["body"].rstrip("\n"))
    out = out.replace(symbol, f"{symbol}_{variant}").replace(namespace, f"{namespace}_{variant}")
    return (f"// VARIANT `{variant}` of {source}: GENERATED by tools/gen_variants.py; edit the source.\n" + out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=KERNELS,
                        help="write here instead of kernels/ (tests diff against it)")
    args = parser.parse_args()
    for source, symbol, namespace in BASES:
        text = (KERNELS / source).read_text()
        for variant in VARIANTS:
            target = args.output_dir / source.replace(".loom", f"_{variant}.loom")
            target.write_text(generate(variant, text, source, symbol, namespace))
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
