# arcface-loom

insightface's `w600k_r50` face recogniser (ArcFace, iResNet-50) written in **Loom**,
AMD's kernel language from [ROCm/hrx-system](https://github.com/ROCm/hrx-system), for
the Radeon 8060S (gfx1151) in a Strix Halo APU. The third sibling of
[dinov3-loom](https://github.com/zacharydenton/dinov3-loom) and
[scrfd-loom](https://github.com/zacharydenton/scrfd-loom); with scrfd-loom it completes
detect -> align -> embed on Loom.

Work in progress; see `docs/notes.md`.
