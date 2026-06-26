# TensorRT/YOLO CI report

This report is derived from the retained per-trial TensorRT/YOLO interference traces.

- bootstrap trials: 10000
- statistic: median
- confidence level: 95%

## yolov8:adaptive
- engine: /home/thor/skim/pqc_encrpyted_fs/artifacts/yolov8n.plan
- duration_s: 5.0
- trials: 3
- pooled sample count: 9361
- median of trial medians: 0.834357 ms (95% CI 0.829176--0.834967)
- pooled median: 0.833343 ms
- trial summaries:
  - trial 0: median 0.834357 ms (95% CI 0.833852--0.834921), p95 2.637804 ms, p99 2.804910 ms, n=3134
  - trial 1: median 0.829176 ms (95% CI 0.828704--0.829704), p95 2.639872 ms, p99 2.853056 ms, n=3143
  - trial 2: median 0.834967 ms (95% CI 0.834351--0.835648), p95 2.640597 ms, p99 2.841459 ms, n=3084

## yolov8:cpu_only
- engine: /home/thor/skim/pqc_encrpyted_fs/artifacts/yolov8n.plan
- duration_s: 5.0
- trials: 3
- pooled sample count: 11914
- median of trial medians: 0.831991 ms (95% CI 0.827463--0.834527)
- pooled median: 0.831705 ms
- trial summaries:
  - trial 0: median 0.834527 ms (95% CI 0.834296--0.834778), p95 0.893872 ms, p99 0.910291 ms, n=3969
  - trial 1: median 0.831991 ms (95% CI 0.831741--0.832231), p95 0.904399 ms, p99 0.920638 ms, n=3958
  - trial 2: median 0.827463 ms (95% CI 0.827194--0.827815), p95 0.890624 ms, p99 0.911423 ms, n=3987

## yolov8:gpu_only
- engine: /home/thor/skim/pqc_encrpyted_fs/artifacts/yolov8n.plan
- duration_s: 5.0
- trials: 3
- pooled sample count: 6402
- median of trial medians: 0.938852 ms (95% CI 0.901667--0.946403)
- pooled median: 0.932292 ms
- trial summaries:
  - trial 0: median 0.938852 ms (95% CI 0.922658--0.953731), p95 4.861509 ms, p99 5.458449 ms, n=2151
  - trial 1: median 0.901667 ms (95% CI 0.888926--0.917472), p95 4.784871 ms, p99 4.950362 ms, n=2109
  - trial 2: median 0.946403 ms (95% CI 0.933518--0.953279), p95 3.003228 ms, p99 3.959701 ms, n=2142

## yolov8:inference_only
- engine: /home/thor/skim/pqc_encrpyted_fs/artifacts/yolov8n.plan
- duration_s: 5.0
- trials: 3
- pooled sample count: 11866
- median of trial medians: 0.837231 ms (95% CI 0.830093--0.843482)
- pooled median: 0.837343 ms
- trial summaries:
  - trial 0: median 0.843482 ms (95% CI 0.843101--0.843857), p95 0.916288 ms, p99 1.081453 ms, n=3930
  - trial 1: median 0.837231 ms (95% CI 0.836852--0.837528), p95 0.898324 ms, p99 0.916031 ms, n=3971
  - trial 2: median 0.830093 ms (95% CI 0.829815--0.830315), p95 0.892933 ms, p99 0.911111 ms, n=3965
