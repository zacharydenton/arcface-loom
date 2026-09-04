"""Drop-in replacement for insightface's ArcFace `w600k_r50` recogniser.

    from arcface_loom import ArcFaceLoom

    model = ArcFaceLoom()
    embedding = model.get(image_bgr, kps)          # (512,) f32, like ArcFaceONNX.get(img, face)
    embeddings = model.get_feat(aligned_crops)      # (n, 512) f32, like ArcFaceONNX.get_feat
    model.close()

``get`` aligns the face with insightface's ``norm_crop`` (``arcface_loom_align``,
vendored) from its five landmarks -- what ``SCRFD.detect`` / scrfd-loom return --
and embeds the 112x112 crop. ``get_feat`` takes crops a caller has already aligned.
The embedding is the un-normalised 512-vector the ONNX graph outputs, as
insightface's ``face.embedding``; ``compute_sim`` normalises, as insightface does.

Importing this module does not initialize HIP. Each model owns an independent
native session; construction loads the kernels and weights, and every call
reuses its GPU allocations. Calls on one object are serialized and may come
from multiple threads.
"""
from __future__ import annotations

import ctypes
import operator
import os
import threading
from pathlib import Path

import numpy as np

from arcface_loom_align import INPUT_SIZE as SIZE, estimate_norm, norm_crop

ROOT = Path(__file__).resolve().parent
EMBEDDING = 512
MAX_BATCH = 64
DEFAULT_MAX_BATCH = 16

_ABI_VERSION = 1
_ERROR_CAPACITY = 4096
_U8Pointer = ctypes.POINTER(ctypes.c_uint8)
_FloatPointer = ctypes.POINTER(ctypes.c_float)


class ArcFaceError(RuntimeError):
    """The native ArcFace runtime could not initialize or complete a call."""


def compute_sim(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """insightface's ``compute_sim``: the cosine between two embeddings."""
    feat1, feat2 = np.asarray(feat1, np.float64).ravel(), np.asarray(feat2, np.float64).ravel()
    return float(feat1 @ feat2 / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


def _path(value: str | os.PathLike[str] | None, environment: str, default: Path) -> Path:
    if value is None:
        value = os.environ.get(environment, default)
    return Path(value).expanduser().resolve()


def _message(error: ctypes.Array[ctypes.c_char], fallback: str) -> str:
    return error.value.decode("utf-8", errors="replace") or fallback


class ArcFaceLoom:
    """A resident, reusable Loom recognition session.

    Calls on one object are serialized and may safely come from multiple Python
    threads. Use separate objects for independent sessions. A session inherited
    across ``fork()`` is deliberately rejected because HIP state is not fork-safe.
    """

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        kernels: str | os.PathLike[str] | None = None,
        library: str | os.PathLike[str] | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        try:
            max_batch = operator.index(max_batch)
        except TypeError as failure:
            raise TypeError("max_batch must be an integer") from failure
        if not 1 <= max_batch <= MAX_BATCH:
            raise ValueError(f"max_batch must be in 1..{MAX_BATCH}, got {max_batch}")

        self.weights = _path(weights, "ARCFACE_LOOM_WEIGHTS", ROOT / "build/weights")
        self.kernels = _path(kernels, "ARCFACE_LOOM_KERNELS", ROOT / "build/kernels")
        self.library = _path(library, "ARCFACE_LOOM_LIBRARY", ROOT / "build/libarcface.so")
        for path, environment, hint, source_path in (
            (self.library, "ARCFACE_LOOM_LIBRARY", "./scripts/build_host.sh", ROOT / "scripts/build_host.sh"),
            (self.kernels, "ARCFACE_LOOM_KERNELS", "./scripts/build_kernels.sh", ROOT / "scripts/build_kernels.sh"),
            (self.weights, "ARCFACE_LOOM_WEIGHTS", "python3 tools/export_weights.py", ROOT / "tools/export_weights.py"),
        ):
            if not path.exists():
                advice = f"set {environment} to its location"
                if source_path.exists():
                    advice += f" or run from the source checkout: {hint}"
                raise FileNotFoundError(f"{path} is missing; {advice}")

        try:
            native = ctypes.CDLL(self.library)
        except OSError as failure:
            raise ArcFaceError(f"cannot load {self.library}: {failure}") from failure
        try:
            self._configure(native)
        except AttributeError as failure:
            raise ArcFaceError(
                f"{self.library} does not expose the ArcFace ABI; rebuild it with ./scripts/build_host.sh"
            ) from failure
        abi = native.arcface_abi_version()
        if abi != _ABI_VERSION:
            raise ArcFaceError(
                f"{self.library} uses ABI {abi}; Python expects ABI {_ABI_VERSION}; "
                "rebuild it with ./scripts/build_host.sh"
            )
        self.size = native.arcface_input_size()
        self.embedding_size = native.arcface_embedding_size()
        if (self.size, self.embedding_size) != (SIZE, EMBEDDING):
            raise ArcFaceError(f"{self.library} is built for {self.size}x{self.size} -> {self.embedding_size}, "
                               f"Python expects {SIZE}x{SIZE} -> {EMBEDDING}")

        # The reusable host buffers come before the native session so a rare
        # NumPy allocation failure cannot strand an already-created GPU session.
        self._input = np.zeros((max_batch, self.size, self.size, 3), np.uint8)
        self._output = np.empty((max_batch, self.embedding_size), np.float32)
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._native = native
        self._handle: ctypes.c_void_p | None = None

        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = native.arcface_create(
            os.fsencode(self.weights), os.fsencode(self.kernels), max_batch,
            ctypes.byref(handle), error, len(error),
        )
        if status:
            raise ArcFaceError(_message(error, f"arcface_create failed with status {status}"))
        if not handle:
            raise ArcFaceError("arcface_create succeeded without returning a session")
        self._handle = handle
        self.max_batch = max_batch
        # insightface's ArcFaceONNX attributes, for callers that read them
        self.input_size = (self.size, self.size)
        self.input_mean, self.input_std = 127.5, 127.5
        self.taskname = "recognition"

    @staticmethod
    def _configure(native: ctypes.CDLL) -> None:
        native.arcface_abi_version.argtypes = []
        native.arcface_abi_version.restype = ctypes.c_uint32
        native.arcface_input_size.argtypes = []
        native.arcface_input_size.restype = ctypes.c_int
        native.arcface_embedding_size.argtypes = []
        native.arcface_embedding_size.restype = ctypes.c_int
        native.arcface_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.arcface_create.restype = ctypes.c_int
        native.arcface_run.argtypes = [
            ctypes.c_void_p,
            _U8Pointer, ctypes.c_size_t, ctypes.c_int,
            _FloatPointer, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.arcface_run.restype = ctypes.c_int
        native.arcface_max_batch.argtypes = [ctypes.c_void_p]
        native.arcface_max_batch.restype = ctypes.c_int
        native.arcface_destroy.argtypes = [ctypes.c_void_p]
        native.arcface_destroy.restype = None

    @property
    def closed(self) -> bool:
        """Whether this model has released its native session."""
        return self._handle is None

    def _ensure_usable(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise ArcFaceError("this ArcFaceLoom session is closed")
        if os.getpid() != self._pid:
            raise ArcFaceError("this ArcFaceLoom session was inherited across fork; create a new model in the child process")
        return self._handle

    def close(self) -> None:
        """Release all GPU allocations and loaded modules; safe to call twice."""
        # Check this before touching the lock. If another thread owned the lock
        # when the process forked, the child's copy can never be acquired because
        # that owner thread no longer exists. HIP state must not be destroyed in
        # the child, so simply invalidate its copied handle.
        if getattr(self, "_pid", os.getpid()) != os.getpid():
            self._handle = None
            return
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            self._handle = None
            if handle is not None:
                self._native.arcface_destroy(handle)

    def __enter__(self) -> ArcFaceLoom:
        self._ensure_usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter teardown, when module globals
            # and the dynamic loader may already be partly dismantled.
            pass

    # --- insightface-compatible API ---------------------------------------------
    def get_feat(self, imgs) -> np.ndarray:
        """Aligned ``(112, 112, 3)`` uint8 BGR crops -- one, a list, or a stacked
        ``(B, 112, 112, 3)`` array -- to ``(B, 512)`` f32 embeddings, ``max_batch``
        crops per GPU call. What ``ArcFaceONNX.get_feat`` returns."""
        stacked = np.asarray(imgs)
        if stacked.ndim == 3:
            stacked = stacked[None]
        expected = (self.size, self.size, 3)
        if stacked.ndim != 4 or stacked.shape[1:] != expected or stacked.dtype != np.uint8:
            raise ValueError(f"expected (B, {self.size}, {self.size}, 3) uint8 BGR crops, got {stacked.shape} {stacked.dtype}")
        batch = stacked.shape[0]
        out = np.empty((batch, self.embedding_size), np.float32)
        # This must precede lock acquisition: an RLock copied while held by a
        # different thread cannot be acquired in the forked child.
        self._ensure_usable()
        with self._lock:
            self._ensure_usable()
            for start in range(0, batch, self.max_batch):
                chunk = stacked[start:start + self.max_batch]
                out[start:start + len(chunk)] = self._run(chunk)
        return out

    def get(self, img: np.ndarray, face) -> np.ndarray:
        """``ArcFaceONNX.get``: align the face from its five landmarks and embed it.
        ``face`` is a ``(5, 2)`` landmark array or anything with a ``.kps``
        attribute (insightface's ``Face``); the latter also gets ``.embedding`` set."""
        kps = getattr(face, "kps", face)
        embedding = self.get_feat(norm_crop(img, np.asarray(kps, np.float32), self.size))[0]
        if hasattr(face, "kps"):
            try:
                face.embedding = embedding
            except (AttributeError, TypeError):
                pass
        return embedding

    def embed(self, img: np.ndarray, kps_list) -> np.ndarray:
        """Many faces of one image: ``(n, 5, 2)`` landmarks to ``(n, 512)``."""
        kps = np.asarray(kps_list, np.float32)
        if kps.ndim != 3 or kps.shape[1:] != (5, 2):
            raise ValueError(f"expected (n, 5, 2) landmarks, got {kps.shape}")
        if len(kps) == 0:
            # Route the no-op through the same lifecycle checks as every other
            # inference call, including closed and fork-inherited sessions.
            return self.get_feat(np.empty((0, self.size, self.size, 3), np.uint8))
        return self.get_feat(np.stack([norm_crop(img, k, self.size) for k in kps]))

    compute_sim = staticmethod(compute_sim)

    def _run(self, chunk: np.ndarray) -> np.ndarray:
        batch = chunk.shape[0]
        handle = self._ensure_usable()
        self._input[:batch] = chunk
        inp, out = self._input[:batch], self._output[:batch]
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = self._native.arcface_run(
            handle, inp.ctypes.data_as(_U8Pointer), inp.nbytes, batch,
            out.ctypes.data_as(_FloatPointer), out.size, error, len(error),
        )
        if status:
            raise ArcFaceError(_message(error, f"arcface_run failed with status {status}"))
        return out.copy()


__all__ = ["ArcFaceLoom", "ArcFaceError", "compute_sim", "norm_crop", "estimate_norm", "SIZE", "EMBEDDING"]
