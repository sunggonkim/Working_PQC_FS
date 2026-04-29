#!/bin/bash
cd /home/thor/skim/pqc_encrpyted_fs
mkdir -p build mnt_secure storage_physical
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ..
bash measure_realistic_scaling.sh > /tmp/real_scaling_output.log 2>&1
echo "BENCHMARK_COMPLETE" >> /tmp/real_scaling_output.log
