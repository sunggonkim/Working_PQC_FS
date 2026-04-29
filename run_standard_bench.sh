#!/bin/bash
set -e

MNT_DIR="/tmp/pqc_mnt"
STORE_DIR="/tmp/pqc_store"

echo "=== Standard Workload Verification ==="

# Cleanup
fusermount -u $MNT_DIR 2>/dev/null || true
rm -rf $STORE_DIR $MNT_DIR
mkdir -p $STORE_DIR $MNT_DIR

# Start FUSE daemon
echo "Starting PQC-FUSE daemon..."
./build/pqc_fuse_gpu $STORE_DIR $MNT_DIR -f > fuse_standard_bench.log 2>&1 &
FUSE_PID=$!
sleep 2

# Verify mount
if ! mountpoint -q $MNT_DIR; then
    echo "FATAL: FUSE mount failed!"
    kill $FUSE_PID 2>/dev/null || true
    exit 1
fi
echo "FUSE mounted OK at $MNT_DIR (PID=$FUSE_PID)"

# Run standard benchmark
python3 standard_io_bench.py $MNT_DIR

echo ""
echo "Shutting down FUSE daemon..."
kill $FUSE_PID 2>/dev/null || true
sleep 1
fusermount -u $MNT_DIR 2>/dev/null || true
rm -rf $STORE_DIR $MNT_DIR
echo "Done."
