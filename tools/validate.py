"""End-to-end: the Loom runner vs onnxruntime and vs insightface's embeddings.

Three gates on the six faces of t1.jpg:
  1. the 512-d embedding of each face vs onnxruntime CPU on the same crop, by cosine;
  2. the same vs the fixture of insightface's own ArcFaceONNX.get() (its own crops,
     which differ from ours by float32-SVD noise in the alignment; test_align.py),
     and the 6x6 similarity matrix the two produce;
  3. batch invariance: the six-crop batch equals six single-crop runs bit for bit.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import reference as R
import test_align as TA
from test_reference import ort_session, cosine

ROOT = Path(__file__).resolve().parent.parent
MIN_COSINE = 0.99995
MAX_SIMILARITY_DELTA = 1e-3


def run_loom(crops: np.ndarray, extra_args: list[str] = ()) -> np.ndarray:
    """(B,112,112,3) BGR uint8 crops -> (B, 512) f32 embeddings, via the CLI."""
    crops = np.ascontiguousarray(crops, dtype=np.uint8)
    if crops.ndim == 3:
        crops = crops[None]
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "crops.bin", Path(tmp) / "embeddings.bin"
        crops.tofile(src)
        subprocess.run([str(ROOT / "host/arcface"), "--weights", str(ROOT / "build/weights"),
                        "--kernels", str(ROOT / "build/kernels"), "--input", str(src),
                        "--output", str(dst), "--batch", str(len(crops))] + list(extra_args),
                       check=True, cwd=ROOT, env=env, capture_output=True)
        return np.fromfile(dst, dtype=np.float32).reshape(len(crops), G.EMBEDDING)


def normed(e: np.ndarray) -> np.ndarray:
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main() -> int:
    import cv2
    ok = True
    fx = TA.fixture()
    img = cv2.imread(str(TA.test_image()))
    crops = TA.crops(img, fx)
    got = run_loom(crops, sys.argv[1:])

    sess = ort_session()
    want = sess.run(None, {sess.get_inputs()[0].name: R.blob_from_bgr_f64(crops).astype(np.float32)})[0]
    for i in range(len(crops)):
        c = cosine(got[i], want[i]); err = np.abs(got[i] - want[i]).max()
        good = c > MIN_COSINE
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} face {i} vs onnxruntime: cosine={c:.7f} max_abs={err:.3e}")

    fixture = np.array([f["embedding"] for f in fx["faces"]], np.float32)
    cos = [cosine(g, w) for g, w in zip(got, fixture)]
    good = min(cos) > MIN_COSINE
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} vs insightface's own embeddings: cosine min={min(cos):.7f}")
    sim_got, sim_want = normed(got) @ normed(got).T, normed(fixture) @ normed(fixture).T
    err = np.abs(sim_got - sim_want).max()
    good = err < MAX_SIMILARITY_DELTA
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} 6x6 similarity matrix vs insightface: max |delta| = {err:.4f}")

    singles = np.concatenate([run_loom(crops[i:i + 1]) for i in range(len(crops))])
    good = np.array_equal(singles, got)
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} batch {len(crops)} equals {len(crops)} batch-1 runs bit for bit")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
