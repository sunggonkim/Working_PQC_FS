# GPU Data-Plane Negative Control

- Overall pass: `True`
- GPU available: `True`
- All CPU/GPU outputs verified: `True`
- Small/mid GPU loss observed: `True`

| blocks | bytes | CPU median ns | GPU median ns | GPU/CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 2750 | 5667222 | 2060.81 |
| 4 | 16384 | 10407 | 6312083 | 606.52 |
| 16 | 65536 | 41009 | 7555305 | 184.24 |
| 64 | 262144 | 166944 | 8559732 | 51.27 |

This is a data-plane negative control for CPU-first AES-GCM placement. It does not claim GPU data-plane offload is always slower, only that the retained small/mid block evidence includes loss cases under the current UMA/CUDA executor.
