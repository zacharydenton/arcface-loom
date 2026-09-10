# HRX migration measurements — 2026-09-10

Radeon 8060S (`gfx1151`), HRX 0.4.0, batch 6. Each measurement uses
10 warmups and 100 samples. Forward measurements alternate graph replay and
direct dispatch and synchronize each sample. End-to-end measurements include
preprocessing, upload, inference and download; image decoding and model setup
are excluded. These are local measurements, not cross-machine performance claims.

| Path | Median ms |
| --- | ---: |
| HRX graph forward | 4.440 |
| HRX direct forward | 4.465 |
| Rust end-to-end | 4.428 |
| Former Python API end-to-end, old compiler | 4.433 |

Full distributions and setup timings: [JSON](benchmark-2026-09-10.json).
Earlier benchmark files describe the former implementation and toolchain.

Rust end-to-end throughput is within 1% of the old API in this run. Graph
replay removes repeated host dispatch setup while retaining the tuned kernels.
