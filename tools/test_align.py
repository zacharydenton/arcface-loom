"""The vendored alignment against insightface's own: the 2x3 affine and the crop
bytes for every face in the fixture, and the blob arithmetic against cv2's."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arcface_loom_align as A
import reference as R

FIXTURE = ROOT / "tools/fixtures/t1_arcface.json"
IMAGE = ROOT / "build/images/t1.jpg"
IMAGE_URL = ("https://raw.githubusercontent.com/deepinsight/insightface/master/"
             "python-package/insightface/data/images/t1.jpg")


def test_image() -> Path:
    if not IMAGE.exists():
        import urllib.request
        IMAGE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, IMAGE)
    return IMAGE


def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def crops(img_bgr: np.ndarray, fx: dict) -> np.ndarray:
    return np.stack([A.norm_crop(img_bgr, np.array(f["kps"], np.float32)) for f in fx["faces"]])


def main() -> int:
    ok = True
    fx = fixture()
    img = cv2.imread(str(test_image()))
    for i, f in enumerate(fx["faces"]):
        kps = np.array(f["kps"], np.float32)
        M = A.estimate_norm(kps)
        m_err = np.abs(M - np.array(f["M"])).max()
        crop = A.norm_crop(img, kps)
        same = hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest() == f["crop_sha256"]
        # insightface runs the Umeyama SVD in float32 (its landmarks' dtype), so the
        # affine depends on the BLAS at the 1e-5 level: bit-identical to the fixture
        # under the numpy that captured it, a few 1e-5 under another. The crop then
        # differs by an intensity level in a few pixels, far below the f16 noise of
        # the network, which test_reference.py grades by cosine.
        good = m_err < 2e-4
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} face {i}: |M - insightface| = {m_err:.1e}, crop bytes "
              f"{'identical' if same else 'differ (float32 SVD, other BLAS)'}")

    # The float64 blob against cv2.dnn.blobFromImages, the arithmetic insightface uses.
    c = crops(img, fx)
    blob = cv2.dnn.blobFromImages(list(c), 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    err = np.abs(blob.astype(np.float64) - R.blob_from_bgr_f64(c)).max()
    good = err < 1e-6
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} blob vs cv2.dnn.blobFromImages: max_abs={err:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
