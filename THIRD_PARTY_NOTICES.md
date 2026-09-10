# Third-party notices

`src/alignment.rs` reproduces two pieces of other people's code so the installed
package can align faces without depending on them:

- `ARCFACE_DST`, `estimate_norm` and `norm_crop` follow
  `insightface/utils/face_align.py` from
  [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/7fadd420c2351d0ffa8cac403421c1a3ed733365/python-package/insightface/utils/face_align.py), MIT License,
  Copyright (c) 2018 Jiankang Deng and Jia Guo.
- `umeyama` follows `skimage/transform/_geometric.py::_umeyama` from
  [scikit-image 0.26.0](https://github.com/scikit-image/scikit-image/blob/v0.26.0/skimage/transform/_geometric.py), BSD-3-Clause License,
  Copyright 2009-2022 the scikit-image team.

The local implementations add input validation and adapt names and signatures.
The applicable license texts are reproduced below. InsightFace's code license
policy is in its [README](https://github.com/deepinsight/insightface#license);
scikit-image's notices are in its
[LICENSE.txt](https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt).


The model this repo runs, `w600k_r50.onnx` from insightface's `buffalo_l` pack, is
distributed by insightface for non-commercial research purposes and is not part of
this repository.

## InsightFace — MIT License

Copyright (c) 2018 Jiankang Deng and Jia Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## scikit-image — BSD-3-Clause License

Copyright 2009-2022 the scikit-image team

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Validation fixture

`tests/fixtures/t1.png` preserves the decoded pixels of InsightFace's
[`t1.jpg`](https://github.com/deepinsight/insightface/blob/7fadd420c2351d0ffa8cac403421c1a3ed733365/python-package/insightface/data/images/t1.jpg).
The accompanying JSON records the existing InsightFace reference results.
These retain the InsightFace MIT attribution above. No model weights are included.
