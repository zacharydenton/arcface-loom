"""Capture insightface's own ArcFace output on t1.jpg, once, as the fixture the
vendored alignment and the whole recogniser are graded against.

Uses insightface's real `face_align.py` (imported by file path from a source
checkout, since the package's top-level import needs a compiled extension) and
its real `get_feat` arithmetic (`blobFromImages` with mean 127.5, std 127.5,
swapRB) on onnxruntime CPU. The five-point landmarks come from scrfd-loom's
fixture of insightface's own `SCRFD.detect()` on the same image.

Run it in an environment that has scikit-image (the one dependency this repo
does not otherwise carry):

  uv run --python 3.12 --with scikit-image --with onnxruntime --with onnx \
         --with opencv-python-headless --with numpy python3 tools/capture_fixture.py

Stores per face: the landmarks, the 2x3 affine, a SHA-256 of the crop bytes and
the 512-d embedding, plus the versions that produced them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "build/images/t1.jpg"
KPS_FIXTURE = Path(os.environ.get("SCRFD_FIXTURE", "~/code/scrfd-loom/tools/fixtures/t1_insightface.json")).expanduser()
FACE_ALIGN = Path(os.environ.get("INSIGHTFACE_FACE_ALIGN",
                                 "~/code/insightface/python-package/insightface/utils/face_align.py")).expanduser()
MODEL = Path(os.environ.get("ARCFACE_ONNX", "~/.insightface/models/buffalo_l/w600k_r50.onnx")).expanduser()
OUT = ROOT / "tools/fixtures/t1_arcface.json"


def main() -> None:
    spec = importlib.util.spec_from_file_location("face_align", FACE_ALIGN)
    face_align = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_align)
    import skimage

    img = cv2.imread(str(IMAGE))
    assert img is not None, IMAGE
    kps_all = np.array(json.loads(KPS_FIXTURE.read_text())["kps"], dtype=np.float32)   # (n, 5, 2)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    faces = []
    for kps in kps_all:
        M = face_align.estimate_norm(kps, 112)                      # exactly ArcFaceONNX.get()
        crop = face_align.norm_crop(img, landmark=kps, image_size=112)
        blob = cv2.dnn.blobFromImages([crop], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = sess.run(None, {input_name: blob})[0][0]
        faces.append({"kps": kps.tolist(), "M": np.asarray(M, np.float64).tolist(),
                      "crop_sha256": hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest(),
                      "embedding": emb.astype(np.float32).tolist()})
    OUT.write_text(json.dumps({
        "image": IMAGE.name, "model": MODEL.name, "input_size": 112,
        "versions": {"onnxruntime": ort.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "scikit-image": skimage.__version__,
                     "face_align": str(FACE_ALIGN)},
        "faces": faces}, indent=1))
    print(f"{len(faces)} faces -> {OUT}")
    e = np.array([f["embedding"] for f in faces])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    print(np.round(e @ e.T, 3))


if __name__ == "__main__":
    main()
