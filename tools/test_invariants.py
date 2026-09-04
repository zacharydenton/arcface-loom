"""Safety checks that must still run when Python assertions are disabled."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import arcface_loom_align as A  # noqa: E402
import export_weights as E      # noqa: E402
import gen_conv as C            # noqa: E402
import gen_launch_table as L     # noqa: E402
import gen_variants as V         # noqa: E402
import graph as G                # noqa: E402


def main() -> int:
    cases = (
        lambda: A.estimate_norm(np.zeros((5, 2), np.float32), 100),
        lambda: E.head_splits(1),
        lambda: C.require(False, "probe"),
        lambda: L.require(False, "probe"),
        lambda: V.require(False, "probe"),
        lambda: G.require(False, "probe"),
    )
    for call in cases:
        try:
            call()
        except (TypeError, ValueError, RuntimeError):
            continue
        print("a safety invariant disappeared under python -O", file=sys.stderr)
        return 1
    print("  PASS graph, export, generator and alignment checks remain active")
    return 0


if __name__ == "__main__":
    sys.exit(main())
