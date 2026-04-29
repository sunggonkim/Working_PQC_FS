#!/bin/bash
set -e

MNT_DIR="/tmp/pqc_mnt"
STORE_DIR="/tmp/pqc_store"

echo "Cleaning up..."
fusermount -u $MNT_DIR 2>/dev/null || true
rm -rf $STORE_DIR $MNT_DIR
mkdir -p $STORE_DIR $MNT_DIR

echo "Starting Q-Learning PQC FUSE Daemon..."
./build/pqc_fuse_gpu $STORE_DIR $MNT_DIR -f > fuse_citadel.log 2>&1 &
FUSE_PID=$!
sleep 2

echo "Starting ROS2 Workload Simulator & I/O Benchmark..."
python3 ros2_workload_sim.py &
SIM_PID=$!

python3 citadel_io_generator.py $MNT_DIR

wait $SIM_PID

echo "Shutting down..."
kill $FUSE_PID || true
fusermount -u $MNT_DIR 2>/dev/null || true
rm -rf $STORE_DIR $MNT_DIR
echo "CITADEL Benchmark Complete."
