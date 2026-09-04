"""Face alignment for ArcFace, exactly as insightface does it, without insightface.

``norm_crop(image_bgr, kps)`` maps the five detected landmarks onto the canonical
ArcFace template with a similarity transform and warps the image to 112x112, the
crop the recogniser was trained on. The template and the procedure are
insightface's ``utils/face_align.py`` (MIT); the similarity estimate is
scikit-image's ``_umeyama`` (BSD-3), reproduced here so the installed package
needs neither. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import numpy as np

INPUT_SIZE = 112

# insightface's `arcface_dst`: the five landmarks (eyes, nose, mouth corners) of
# the canonical 112x112 ArcFace face.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """Least-squares similarity transform mapping src to dst (Umeyama 1991),
    step for step scikit-image's ``_umeyama`` -- including its arithmetic in the
    caller's dtype (float32 from insightface), so the crops come out byte-identical.
    Returns the (dim+1)x(dim+1) homogeneous matrix."""
    src = np.asarray(src)
    dst = np.asarray(dst)
    if src.ndim != 2 or dst.shape != src.shape:
        raise ValueError(f"src and dst must have the same (points, dimensions) shape, got {src.shape} and {dst.shape}")
    if src.shape[0] == 0 or src.shape[1] == 0:
        raise ValueError("src and dst must contain at least one point and one dimension")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("src and dst coordinates must be finite")
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    tol = S.max() * np.max(A.shape) * np.finfo(float).eps
    rank = np.count_nonzero(S > tol)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d) if estimate_scale else 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(kps: np.ndarray, image_size: int = INPUT_SIZE) -> np.ndarray:
    """insightface's estimate_norm: the 2x3 affine taking the (5, 2) landmarks to
    the template at image_size (a multiple of 112, or of 128 with its x offset)."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        raise ValueError(f"expected (5, 2) landmarks, got {kps.shape}")
    if not isinstance(image_size, (int, np.integer)) or isinstance(image_size, (bool, np.bool_)):
        raise TypeError(f"image_size must be an integer, got {type(image_size).__name__}")
    image_size = int(image_size)
    if image_size <= 0 or (image_size % 112 != 0 and image_size % 128 != 0):
        raise ValueError(f"image_size must be a positive multiple of 112 or 128, got {image_size}")
    if image_size % 112 == 0:
        ratio, diff_x = float(image_size) / 112.0, 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    transform = umeyama(kps, dst, True)[0:2, :]
    if not np.isfinite(transform).all():
        raise ValueError("landmarks do not define a finite similarity transform")
    return transform


def norm_crop(image_bgr: np.ndarray, kps: np.ndarray, image_size: int = INPUT_SIZE) -> np.ndarray:
    """The aligned (image_size, image_size, 3) BGR uint8 crop insightface feeds the
    recogniser: cv2.warpAffine, bilinear, black outside the image."""
    import cv2
    M = estimate_norm(kps, image_size)
    return cv2.warpAffine(image_bgr, M, (image_size, image_size), borderValue=0.0)


__all__ = ["ARCFACE_DST", "INPUT_SIZE", "estimate_norm", "norm_crop", "umeyama"]
