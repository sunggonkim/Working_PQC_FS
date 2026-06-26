# UVM proxy report

This report packages the existing managed-memory smoke evidence and the emulated semantic-gap projection.

## Managed-memory smoke
- self_test_rc: 0
- managed_alloc_bytes: 16777216
- managed_free_bytes: 16777216
- prefetch_to_device_bytes: 16777216
- prefetch_to_host_bytes: 16777216
- prefetch_device_calls: 1
- prefetch_host_calls: 1

## Semantic gap projection
- Sequential I/O (fscrypt): page_faults=2560, page_migrations=1280, read_ahead_waste_pct=0.2, provenance=emulated_projection_jetson_orin_nano (eBPF failed: debugfs missing)
- Lattice PQC NTT (UVM): page_faults=655360, page_migrations=327680, read_ahead_waste_pct=91.4, provenance=emulated_projection_jetson_orin_nano (eBPF failed: debugfs missing)

## Interpretation
- managed-memory smoke: proxy evidence only, not DMA or migration suppression proof
- semantic-gap projection: emulated because the trace backend was unavailable
