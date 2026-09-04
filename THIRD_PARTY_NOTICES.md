# Third-party notices

`arcface_loom_align.py` reproduces two pieces of other people's code so the installed
package can align faces without depending on them:

- `ARCFACE_DST`, `estimate_norm` and `norm_crop` follow
  `insightface/utils/face_align.py` from
  [deepinsight/insightface](https://github.com/deepinsight/insightface), MIT License,
  Copyright (c) 2018 Jiankang Deng and Jia Guo.
- `umeyama` follows `skimage/transform/_geometric.py::_umeyama` from
  [scikit-image](https://github.com/scikit-image/scikit-image), BSD-3-Clause License,
  Copyright (c) the scikit-image team.

`tools/capture_fixture.py` imports insightface's `face_align.py` from a source checkout
at capture time only; nothing from it ships.

The model this repo runs, `w600k_r50.onnx` from insightface's `buffalo_l` pack, is
distributed by insightface for non-commercial research purposes and is not part of
this repository.
