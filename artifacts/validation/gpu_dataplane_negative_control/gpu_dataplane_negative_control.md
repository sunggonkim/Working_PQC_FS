# GPU Data-Plane Negative Control

- Overall pass: `True`
- GPU available: `True`
- All CPU/GPU outputs verified: `True`
- Small/mid GPU loss observed: `True`

| blocks | bytes | CPU median ns | GPU median ns | GPU/CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 2740 | 5636472 | 2057.11 |
| 4 | 16384 | 10454 | 6290306 | 601.71 |
| 16 | 65536 | 41241 | 7298519 | 176.97 |
| 64 | 262144 | 164417 | 8100473 | 49.27 |
| 256 | 1048576 | 658463 | 13220889 | 20.08 |

This is a data-plane negative control for CPU-first AES-GCM placement. It does not claim GPU data-plane offload is always slower, only that the retained small/mid block evidence includes loss cases under the current UMA/CUDA executor.
