#!/usr/bin/env bash
# The one test command. Formats, checks the generated files against their
# generators, exports, builds every kernel and the host, runs every unit test
# against a float64 reference on the real GPU, checks the runner's error paths,
# then validates the whole recogniser against onnxruntime and insightface's
# embeddings.
#
#   scripts/test.sh          everything
#   scripts/test.sh --quick  skip the onnxruntime/insightface comparisons
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

quick=0
[ "${1:-}" = "--quick" ] && quick=1
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
export tmpdir

status=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi
}

step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'
step "generated kernels match their generators" bash -c '
  python3 tools/gen_conv.py --output-dir "$tmpdir" >/dev/null &&
  python3 tools/gen_variants.py --output-dir "$tmpdir" >/dev/null &&
  for f in conv3x3_f16_wmma conv3x3_f16_wmma_prelu conv3x3_f16_wmma_bnprelu conv3x3_f16_wmma_add \
           conv3x3_n128_f16_wmma conv3x3_n128_f16_wmma_prelu conv3x3_n128_f16_wmma_bnprelu conv3x3_n128_f16_wmma_add; do
    "$LOOM_FORMAT" --in-place "$tmpdir/$f.loom" >/dev/null && cmp -s "$tmpdir/$f.loom" "kernels/$f.loom" || { echo "  $f differs"; exit 1; }
  done'
step "launch table matches the graph" bash -c '
  cp host/graph_table.inc "$tmpdir/table.inc" && cp scripts/kernels.generated "$tmpdir/kernels.generated" &&
  python3 tools/gen_launch_table.py >/dev/null &&
  cmp -s host/graph_table.inc "$tmpdir/table.inc" && cmp -s scripts/kernels.generated "$tmpdir/kernels.generated"'
step "safety checks survive python -O" python3 -O tools/test_invariants.py
step "export folds are exact (float64)" python3 tools/test_export_fold.py
step "export weights" python3 tools/export_weights.py
step "build kernels" ./scripts/build_kernels.sh
step "build host programs" ./scripts/build_host.sh
step "native errors drain uploads and invalidate failed sessions" python3 tools/test_native_errors.py

step "uint8 bgr -> nhwc f16"     python3 tools/test_convert.py
step "im2col"                    python3 tools/test_im2col.py
step "explicit conv (day one)"   python3 tools/test_conv_explicit.py
step "conv3x3 + variants"        python3 tools/test_conv3x3.py
step "head: split-K + reduce"    python3 tools/test_head.py

step "runner rejects bad input" bash -c '
  rejects() {   # <phrase> <command...>: must exit 64 AND print the phrase
    local want="$1"; shift
    local out; out=$("$@" 2>&1); local rc=$?
    if [ "$rc" != 64 ]; then echo "  expected exit 64, got $rc: $*"; return 1; fi
    if ! grep -q -- "$want" <<<"$out"; then echo "  missing \"$want\" in: $out"; return 1; fi
  }
  printf short > "$tmpdir/short.bin"
  python3 -c "import numpy as np; np.zeros(2*112*112*3, np.uint8).tofile(\"$tmpdir/two.bin\")"
  rejects "needs a value"     ./host/arcface --batch                                   &&
  rejects "must be 1"         ./host/arcface --input "$tmpdir/two.bin" --batch 0       &&
  rejects "input is required" ./host/arcface                                           &&
  rejects "not a multiple"    ./host/arcface --input "$tmpdir/short.bin"               &&
  rejects "holds 2 crops"     ./host/arcface --input "$tmpdir/two.bin" --batch 3'
step "session rejects a lying manifest" bash -c '
  cp -r build/weights "$tmpdir/badw" && sed -i "s/^c03 \([0-9]*\) [0-9]*$/c03 \1 1/" "$tmpdir/badw/manifest_f16.txt" &&
  python3 -c "import numpy as np; np.zeros(112*112*3, np.uint8).tofile(\"$tmpdir/one.bin\")" &&
  out=$(./host/arcface --weights "$tmpdir/badw" --input "$tmpdir/one.bin" 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -q "the kernel expects" <<<"$out"'

if [ "$quick" = 0 ]; then
  step "alignment vs insightface"                      python3 tools/test_align.py
  step "reference vs onnxruntime"                      python3 tools/test_reference.py
  step "end to end vs onnxruntime and insightface"     python3 tools/validate.py
  step "python api: lifecycle, ABI errors, batches"    python3 tools/test_python_api.py
fi

printf '\n'
[ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
