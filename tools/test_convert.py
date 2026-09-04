"""hwc_u8_to_nhwc_f16 vs cv2.dnn.blobFromImages: the aligned BGR uint8 crop ->
normalised RGB NHWC f16 padded to 8 channels, graded against the arithmetic
insightface's get_feat uses (mean 127.5, scale 1/127.5, swapRB)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

P = "arcface.hwc_u8_to_nhwc_f16"
SYMBOL = "arcface_hwc_u8_to_nhwc_f16"


def expected(image: np.ndarray) -> np.ndarray:
    """cv2's blob, [B,3,S,S] f32 RGB, relaid to NHWC and truncated to f16; channels 3..7 zero."""
    import cv2
    batch, size = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImages(list(image), 1.0 / 127.5, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    out = np.zeros((batch, size, size, 8), np.float16)
    out[..., :3] = blob.transpose(0, 2, 3, 1).astype(np.float16)
    return out


def main() -> int:
    ok = True
    rng = np.random.default_rng(3)
    with workdir() as tmp:
        tmp = Path(tmp)
        for size, batch in ((16, 2), (112, 3)):
            hsaco = tmp / f"cvt_{size}.hsaco"
            compile_kernel(ROOT / "kernels/hwc_u8_to_nhwc_f16.loom", SYMBOL, {f"{P}.size": size}, hsaco)
            image = rng.integers(0, 256, (batch, size, size, 3), dtype=np.uint8)
            image[0, 0, 0] = (0, 255, 128)            # the extremes must survive too
            image[0, 0, 1] = (127, 128, 1)
            rows = batch * size
            (out,), timing = launch(hsaco, SYMBOL, (rows, 1, 1), (256, 1, 1),
                                    [("i32", rows), ("in_u8", image),
                                     ("out_f16", ((batch * size * size, 8), np.float16))],
                                    tmp, repeat=20)
            want = expected(image).reshape(-1, 8)
            # (x - 127.5) * (1/127.5) rounds differently in f32 than cv2's double
            # scale on a few values; one f16 ulp (2^-11 at |x| < 1) covers it.
            ok &= report(f"size={size} batch={batch} ({timing['per_launch_us']:7.2f} us, "
                         f"{batch * size * size * 3 / (timing['per_launch_us'] * 1e-6) / 1e9:5.1f} GB/s in)",
                         out, want, atol=2 ** -11, rtol=0)
            exact = np.array_equal(out, want)
            print(f"      {'bit-exact' if exact else f'{np.count_nonzero(out != want)} of {out.size} values differ by one ulp'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
