"""The float64 reference vs onnxruntime (CPU, f32) on the aligned faces of t1.jpg.

This is the oracle's own test: everything downstream is graded against
reference.py, so it has to agree with the deployed graph first. It also checks
that ORT's embeddings match the insightface fixture, which pins the whole
preprocessing chain (alignment, blob) to production.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import reference as R
import test_align as TA


def ort_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3          # the graph declares [1, 512]; batching is fine, ORT warns
    return ort.InferenceSession(str(G.model_path()), opts, providers=["CPUExecutionProvider"])


def cosine(a, b) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    import cv2
    ok = True
    graph = G.load()
    sess = ort_session()
    input_name = sess.get_inputs()[0].name

    fx = TA.fixture()
    img = cv2.imread(str(TA.test_image()))
    crops = TA.crops(img, fx)                                  # (6, 112, 112, 3) BGR uint8
    x = R.blob_from_bgr_f64(crops).astype(np.float32)          # [6,3,112,112] RGB
    want_fixture = np.array([f["embedding"] for f in fx["faces"]], np.float32)

    got = sess.run(None, {input_name: x})[0]
    one = np.concatenate([sess.run(None, {input_name: x[i:i + 1]})[0] for i in range(len(x))])
    batch_same = np.array_equal(got, one)
    ok &= batch_same
    print(f"  {'PASS' if batch_same else 'FAIL'} onnxruntime batch {len(x)} equals six batch-1 runs")

    # Our crops vs insightface's differ by float32-SVD noise in the affine (see
    # test_align.py); the embeddings of the same graph on them agree to ~1e-5 cosine.
    cos = [cosine(g, w) for g, w in zip(got, want_fixture)]
    good = min(cos) > 0.9999
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} onnxruntime on our crops vs insightface fixture: "
          f"cosine min={min(cos):.6f} max_abs={np.abs(got - want_fixture).max():.2e}")

    t = time.perf_counter()
    ref = R.forward(graph, x.astype(np.float64))
    print(f"  reference forward ({len(x)} faces): {time.perf_counter() - t:.1f} s")
    for i in range(len(x)):
        err = np.abs(got[i].astype(np.float64) - ref[i])
        scale = np.abs(ref[i]).max() + 1e-6
        rel = err.max() / scale
        good = rel < 1e-4
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} face {i}: reference vs onnxruntime max_abs={err.max():.2e} "
              f"rel_to_max={rel:.2e} cosine={cosine(got[i], ref[i]):.9f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
