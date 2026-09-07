"""Inject HIP failures after a pinned upload; verify drain, retry and invalidation.

The interposer is compiled in a temporary directory and loaded into a child
process, so its deliberate faults never affect another test or library user.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def child() -> None:
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from arcface_loom import ArcFaceLoom, ArcFaceError

    probe = C.CDLL(None)
    probe.arcface_test_arm.argtypes = [C.c_int]
    probe.arcface_test_arm.restype = None
    probe.arcface_test_count.argtypes = [C.c_int]
    probe.arcface_test_count.restype = C.c_int
    counts = lambda: tuple(probe.arcface_test_count(i) for i in range(3))

    for drain_error in (0, 1):
        with ArcFaceLoom(max_batch=1) as model:
            crop = np.zeros((1, 112, 112, 3), np.uint8)
            expected = model.get_feat(crop)
            native = model._native
            native.hipHostMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint]
            native.hipHostMalloc.restype = C.c_int
            native.hipHostFree.argtypes = [C.c_void_p]
            native.hipHostFree.restype = C.c_int
            pinned = C.c_void_p()
            check(native.hipHostMalloc(C.byref(pinned), crop.nbytes, 0) == 0, "pinned allocation failed")
            try:
                C.memmove(pinned, crop.ctypes.data, crop.nbytes)
                output = np.full((1, 512), 123.0, np.float32)
                error = C.create_string_buffer(4096)
                probe.arcface_test_arm(drain_error)
                status = native.arcface_run(
                    model._handle, C.cast(pinned, C.POINTER(C.c_uint8)), crop.nbytes, 1,
                    output.ctypes.data_as(C.POINTER(C.c_float)), output.size, error, len(error),
                )
                check(status == 1 and b"hipModuleLaunchKernel" in error.value,
                      "original launch error was not preserved")
                check(counts() == (1, 1, 1), f"upload was not drained before error return: {counts()}")
                check(np.all(output == 123.0), "failed run changed the caller's output")
                if drain_error:
                    try:
                        model.get_feat(crop)
                    except ArcFaceError as failure:
                        check("unusable" in str(failure), str(failure))
                    else:
                        raise AssertionError("session accepted a run after failed recovery")
                    check(counts() == (1, 1, 1), "unusable session submitted more GPU work")
                    print("  PASS failed recovery disables the session before further GPU work")
                else:
                    check(np.array_equal(model.get_feat(crop), expected), "retry differs after recovery")
                    print("  PASS launch failure drains pinned upload, preserves output, and permits retry")
            finally:
                check(native.hipHostFree(pinned) == 0, "pinned free failed")


def main() -> None:
    if sys.argv[1:] == ["--child"]:
        child()
        return
    hipcc = os.environ.get("HIPCC", str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"))
    with tempfile.TemporaryDirectory(prefix="arcface-fault-test-") as directory:
        library = Path(directory) / "hip_fault_injection.so"
        subprocess.run([hipcc, "-shared", "-fPIC", "-O2", "-Wall", "-Werror",
                        str(ROOT / "tools/hip_fault_injection.cpp"), "-o", str(library), "-ldl"], check=True)
        env = os.environ.copy()
        env["LD_PRELOAD"] = str(library) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, check=True)


if __name__ == "__main__":
    main()
