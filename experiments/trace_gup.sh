#!/bin/bash
# trace_gup.sh
# Development-only helper that emits a trace script for GUP/pinning inspection.
# It does not by itself prove any storage-DMA claim.

cat << 'EOF' > trace_gup.bt
kprobe:get_user_pages_fast
/comm == "repro"/
{
    printf("get_user_pages_fast called by %s\n", comm);
    @gup_calls[comm] = count();
}

kprobe:get_user_pages
/comm == "repro"/
{
    printf("get_user_pages called by %s\n", comm);
    @gup_calls[comm] = count();
}

kprobe:nvidia_p2p_get_pages
/comm == "repro"/
{
    printf("nvidia_p2p_get_pages called by %s\n", comm);
    @p2p_calls[comm] = count();
}
EOF

echo "Tracing helper generated: trace_gup.bt"
echo "Run it only on a root-capable system where kprobes are available."
