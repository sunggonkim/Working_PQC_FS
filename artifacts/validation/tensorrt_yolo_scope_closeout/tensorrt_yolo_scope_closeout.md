# TensorRT/YOLO Scope Closeout

- overall_pass: `True`
- verdict: `trace-scoped-non-claim`

## Conditions
- ci_report_complete: `True`
- primary_trace_complete: `True`
- all_retained_traces_complete: `True`
- cupti_bridge_complete: `True`
- paper_scope_phrases_present: `True`
- claim_scan_passes: `True`

## Retained Trace Boundary
- `artifacts/results/motivation/tensorrt_interference.json`: complete=`True`, modes=yolov8:adaptive, yolov8:cpu_only, yolov8:gpu_only, yolov8:inference_only
  - yolov8:adaptive: trials=3, samples=9361, median p99=2.8414591774344444 ms
  - yolov8:cpu_only: trials=3, samples=11914, median p99=0.9114225022494793 ms
  - yolov8:gpu_only: trials=3, samples=6402, median p99=4.9503616988658905 ms
  - yolov8:inference_only: trials=3, samples=11866, median p99=0.9160312823951244 ms
- `artifacts/results/qos/m6_tradeoff_yolov8_adaptive_tight/tensorrt_interference.json`: complete=`True`, modes=yolov8:adaptive, yolov8:cpu_only, yolov8:gpu_only, yolov8:inference_only
  - yolov8:adaptive: trials=1, samples=4006, median p99=0.9072899352759123 ms
  - yolov8:cpu_only: trials=1, samples=3998, median p99=0.9024558030068874 ms
  - yolov8:gpu_only: trials=1, samples=3049, median p99=1.1102429032325745 ms
  - yolov8:inference_only: trials=1, samples=3932, median p99=1.0337962210178375 ms
- `artifacts/results/qos/m6_tradeoff_yolov8_elasticcontend/tensorrt_interference.json`: complete=`True`, modes=yolov8:adaptive, yolov8:cpu_only, yolov8:gpu_only, yolov8:inference_only
  - yolov8:adaptive: trials=1, samples=2702, median p99=4.937501326203346 ms
  - yolov8:cpu_only: trials=1, samples=4767, median p99=0.9117248095571995 ms
  - yolov8:gpu_only: trials=1, samples=2706, median p99=4.928612848743796 ms
  - yolov8:inference_only: trials=1, samples=4745, median p99=1.0413509234786034 ms

## Interpretation
- The TensorRT/YOLO rows remain interference/prototype evidence.
- The CUPTI bridge is same-run telemetry-to-mounted-FUSE-throttle wiring evidence.
- The closeout does not permit a foreground AI/TensorRT p99 recovery claim.

## Claim Scan
- candidates: `19`
- unguarded: `0`
