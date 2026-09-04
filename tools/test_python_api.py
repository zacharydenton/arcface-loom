"""Lifecycle, error-boundary and correctness tests for the resident Python API.

Correctness is graded against the CLI on the same crops (bit for bit), against
insightface's own embeddings (the fixture) through get() from landmarks, and
across chunking, threads and a second session."""
from __future__ import annotations

import ctypes
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from arcface_loom import ArcFaceError, ArcFaceLoom, compute_sim   # noqa: E402
import test_align as TA                                            # noqa: E402
from validate import MIN_COSINE, run_loom                          # noqa: E402


def require_raises(kind, phrase: str, call) -> None:
    try:
        call()
    except kind as failure:
        assert phrase in str(failure), str(failure)
    else:
        raise AssertionError(f"expected {kind.__name__} containing {phrase!r}")


def main() -> int:
    import cv2
    ok = True

    def check(name: str, good: bool) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {name}")

    require_raises(ValueError, "1..64", lambda: ArcFaceLoom(max_batch=65))
    require_raises(TypeError, "integer", lambda: ArcFaceLoom(max_batch=2.5))
    # Native initialization failures return through the ABI, never exit().
    with tempfile.TemporaryDirectory() as empty:
        require_raises(ArcFaceError, "manifest", lambda: ArcFaceLoom(weights=empty, max_batch=1))
    with tempfile.TemporaryDirectory() as empty:
        require_raises(ArcFaceError, "hipModuleLoad", lambda: ArcFaceLoom(kernels=empty, max_batch=1))
    check("construction errors: ranges, types, native failures through the ABI", True)

    fx = TA.fixture()
    img = cv2.imread(str(TA.test_image()))
    crops = TA.crops(img, fx)
    kps = np.array([f["kps"] for f in fx["faces"]], np.float32)
    fixture = np.array([f["embedding"] for f in fx["faces"]], np.float32)
    cli = run_loom(crops)

    with ArcFaceLoom(max_batch=4) as model:
        assert model._native.arcface_max_batch(model._handle) == 4
        feats = model.get_feat(crops)                                 # 6 crops through a 4-crop session: two calls
        check("get_feat on a stacked array == the CLI, bit for bit, across chunking", np.array_equal(feats, cli))
        check("get_feat on a list of crops == stacked", np.array_equal(model.get_feat(list(crops)), feats))
        check("get_feat on one crop returns (1, 512)", model.get_feat(crops[0]).shape == (1, 512)
              and np.array_equal(model.get_feat(crops[0])[0], feats[0]))
        cos = [compute_sim(model.get(img, k), f) for k, f in zip(kps, fixture)]
        check(f"get(img, kps) vs insightface's embeddings: cosine min={min(cos):.6f}", min(cos) > MIN_COSINE)

        class Face:
            def __init__(self, k): self.kps = k
        face = Face(kps[0])
        e = model.get(img, face)
        check("get(img, face) sets face.embedding like insightface", np.array_equal(face.embedding, e) and np.array_equal(e, feats[0]))
        check("embed(img, kps) == get_feat on norm_crop", np.array_equal(model.embed(img, kps), feats))
        check("embed with no faces returns (0, 512)", model.embed(img, np.zeros((0, 5, 2))).shape == (0, 512))
        first = model.get_feat(crops[:2]); saved = first.copy()
        model.get_feat(crops[2:4])
        check("a later call does not mutate an earlier result", np.array_equal(first, saved))
        require_raises(ValueError, "uint8", lambda: model.get_feat(crops.astype(np.float32)))
        require_raises(ValueError, "112, 112", lambda: model.get_feat(np.zeros((2, 64, 64, 3), np.uint8)))
        require_raises(ValueError, "(n, 5, 2)", lambda: model.embed(img, np.zeros((3, 4, 2))))

        # The ABI rejects wrong sizes before launching anything.
        error = ctypes.create_string_buffer(1024)
        u8, f32 = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float)
        inp, out = model._input[:1], model._output[:1]
        args = lambda nb, batch, ne: (model._handle, inp.ctypes.data_as(u8), nb, batch,
                                      out.ctypes.data_as(f32), ne, error, len(error))
        status = model._native.arcface_run(*args(inp.nbytes - 1, 1, out.size))
        check("ABI rejects a short input", status == 64 and b"input has" in error.value)
        status = model._native.arcface_run(*args(inp.nbytes, 1, out.size - 1))
        check("ABI rejects a misshapen output", status == 64 and b"embeddings has" in error.value)
        status = model._native.arcface_run(*args(inp.nbytes, 5, out.size))
        check("ABI rejects a batch above max_batch", status == 64 and b"batch must be" in error.value)

        # The Python lock protects the shared buffers while ctypes drops the GIL.
        with ThreadPoolExecutor(max_workers=3) as pool:
            threaded = list(pool.map(lambda c: model.get_feat(c[None])[0], crops))
        check("threads see the same results as sequential calls", np.array_equal(np.stack(threaded), feats))

        with ArcFaceLoom(max_batch=1) as other:
            check("a second session agrees", np.array_equal(other.get_feat(crops), feats))
            with ThreadPoolExecutor(max_workers=2) as pool:
                left = pool.submit(model.get_feat, crops[:2])
                right = pool.submit(other.get_feat, crops[2:4])
                check("independent session streams can run concurrently",
                      np.array_equal(left.result(), feats[:2]) and np.array_equal(right.result(), feats[2:4]))
        check("the first session survives the second's close", np.array_equal(model.get_feat(crops), feats))

    assert model.closed
    model.close()
    require_raises(ArcFaceError, "closed", lambda: model.get_feat(crops))
    require_raises(ArcFaceError, "closed", lambda: model.embed(img, np.empty((0, 5, 2), np.float32)))
    check("closed sessions refuse calls, including empty embed", True)

    # A fork-inherited object must reject the call before entering its copied
    # lock. Such a lock can be permanently held by a parent thread that does not
    # exist in the child; this sentinel makes the ordering deterministic without
    # forking after HIP has initialized.
    class MustNotLock:
        def __enter__(self):
            raise AssertionError("fork guard tried to acquire the inherited lock")
        def __exit__(self, *args):
            return False

    inherited = ArcFaceLoom.__new__(ArcFaceLoom)
    inherited.size, inherited.embedding_size, inherited.max_batch = 112, 512, 1
    inherited._input = np.empty((1, 112, 112, 3), np.uint8)
    inherited._output = np.empty((1, 512), np.float32)
    inherited._handle = ctypes.c_void_p(1)
    inherited._pid = -1
    inherited._lock = MustNotLock()
    require_raises(ArcFaceError, "fork", lambda: inherited.get_feat(crops[:1]))
    inherited.close()
    check("fork-inherited sessions reject without acquiring a possibly orphaned lock", inherited.closed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
