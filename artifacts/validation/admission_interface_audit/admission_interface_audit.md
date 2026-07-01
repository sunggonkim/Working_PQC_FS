# Admission Interface Audit

- Overall pass: `true`
- Scope: producer-facing slack/deadline interface only; not an end-to-end QoS result.

## Cases
- `slack_available_gpu` acceptable=`true`, target=`GPU`, decision_reason=`4`, deferral_reason=`0`, deadline_ns=`10000000`, batch_age_ns=`0`, slack_age_ns=`8408`, stale=`False`
- `no_slack_cpu` acceptable=`true`, target=`CPU`, decision_reason=`2`, deferral_reason=`2`, deadline_ns=`10000000`, batch_age_ns=`0`, slack_age_ns=`6342`, stale=`False`
- `deadline_elapsed_cpu` acceptable=`true`, target=`CPU`, decision_reason=`8`, deferral_reason=`8`, deadline_ns=`10000000`, batch_age_ns=`20000000`, slack_age_ns=`0`, stale=`False`
- `stale_slack_cpu` acceptable=`true`, target=`CPU`, decision_reason=`128`, deferral_reason=`128`, deadline_ns=`10000000`, batch_age_ns=`0`, slack_age_ns=`2071028`, stale=`True`

## Interface Contract
- Deadline source: producer-supplied relative deadline/slack in nanoseconds.
- Timestamp domain: trace timestamp uses CLOCK_REALTIME; batch age and slack age use CLOCK_MONOTONIC-relative nanoseconds.
- Clock synchronization: no cross-process synchronization is assumed for relative age/deadline fields.
- Stale-sample behavior: producer slack older than PQC_PRODUCER_SLACK_STALE_NS routes to CPU with STALE_TELEMETRY.
- No-slack fallback: zero supplied slack routes to CPU with AI_QOS_EXHAUSTED.
- Safety default: admission initializes and falls back to CPU unless all GPU gates pass.
