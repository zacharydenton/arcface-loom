"""Interleaved benchmark: the Python API end to end vs onnxruntime+MIGraphX.

What is timed for this repo is ArcFaceLoom.get_feat on aligned crops: upload,
the network, download, i.e. what a caller pays after alignment. onnxruntime's
MIGraphX provider runs the same graph on the same iGPU; unlike scrfd's, this
graph accepts a batch (it only *declares* [1, 512]), so MIGraphX is timed at
batch 1 and at the largest Loom batch, each compiled once (100-150 s per shape).
Rounds are interleaved and the best of N kept per configuration, the honest
protocol under contention; the CPU load is recorded next to the numbers.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import graph as G
import reference as R
import test_align as TA
from arcface_loom import ArcFaceLoom

ROUNDS = 3
BATCHES = (1, 8, 16, 32)
MIGRAPHX_BATCHES = (1, 16)


def loom_img_per_s(model: ArcFaceLoom, crops: np.ndarray, batch: int, total: int = 256) -> float:
    chunk = crops[:batch]
    model.get_feat(chunk)
    calls = max(2, total // batch)
    t = time.perf_counter()
    for _ in range(calls):
        model.get_feat(chunk)
    return calls * batch / (time.perf_counter() - t)


_sessions: dict[int, object] = {}


def migraphx_ms(blob: np.ndarray, steps: int = 30) -> float:
    import onnxruntime as ort
    batch = blob.shape[0]
    if batch not in _sessions:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        t = time.perf_counter()
        sess = ort.InferenceSession(str(G.model_path()), opts,
                                    providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])
        assert sess.get_providers()[0] == "MIGraphXExecutionProvider", sess.get_providers()
        sess.run(None, {sess.get_inputs()[0].name: blob})
        print(f"  MIGraphX compiled batch {batch} in {time.perf_counter() - t:.0f} s", flush=True)
        _sessions[batch] = sess
    sess = _sessions[batch]
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: blob})
    t = time.perf_counter()
    for _ in range(steps):
        sess.run(None, {name: blob})
    return (time.perf_counter() - t) / steps * 1000


def main() -> None:
    import cv2
    img = cv2.imread(str(TA.test_image()))
    six = TA.crops(img, TA.fixture())
    # distinct crops so nothing can be cached across the batch
    crops = np.concatenate([six, six[:, ::-1], six[:, :, ::-1], six[::-1, ::-1]] * 2)[:max(BATCHES)]
    crops = np.ascontiguousarray(crops)
    blobs = {b: R.blob_from_bgr_f64(crops[:b]).astype(np.float32) for b in MIGRAPHX_BATCHES}
    model = ArcFaceLoom(max_batch=max(BATCHES))

    load = os.getloadavg()[0]
    print(f"w600k_r50 (ArcFace iResNet-50), aligned 112x112 crops -- gfx1151; "
          f"CPU load average {load:.1f}; best of {ROUNDS} interleaved rounds\n")
    best: dict[str, float] = {}
    for r in range(ROUNDS):
        for b in BATCHES:
            key = f"arcface-loom get_feat (batch {b})"
            best[key] = max(best.get(key, 0.0), loom_img_per_s(model, crops, b))
        for b in MIGRAPHX_BATCHES:
            ms = migraphx_ms(blobs[b])
            key = f"onnxruntime MIGraphX (batch {b})"
            best[key] = max(best.get(key, 0.0), b * 1000 / ms)
        print(f"  round {r + 1}/{ROUNDS} done", flush=True)
    model.close()

    print(f"\n{'configuration':<44s} {'img/s':>8s} {'ms/img':>8s}   vs MIGraphX b1")
    ref = best["onnxruntime MIGraphX (batch 1)"]
    for name, v in sorted(best.items(), key=lambda kv: -kv[1]):
        marker = "  <-- this repo" if name.startswith("arcface-loom") else ""
        print(f"{name:<44s} {v:8.1f} {1000 / v:8.3f}   {v / ref:5.2f}x{marker}")


if __name__ == "__main__":
    main()
