#!/bin/bash
# Timing campaign on a rented/dedicated GPU (e.g. Colab A100). Run from the unpacked bundle root:
#   bash scripts/colab_run.sh [reps]            # default 2 repetitions of U3 U4 U5 A1 A2 A3
# Writes runs/timing_<gpu>/ and a results tarball (no VTK files) next to it.
set -eu
REPS=${1:-2}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')
OUT=runs/timing_$GPU
mkdir -p "$OUT" build
{
  date; echo "--- GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.max.sm --format=csv
  echo "--- CPU"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'; echo "nproc $(nproc)"
  echo "--- toolchain"; nvcc --version | tail -2; gcc --version | head -1
} | tee "$OUT/hardware.txt"
nvcc -O3 -arch=native -std=c++17 -Xcompiler -Wall,-Wextra,-fopenmp src/cylinder_fem_amr.cu -o build/cylinder_fem_amr
echo "build ok"
mkdir -p "$OUT/smoke" && (cd "$OUT/smoke" && ../../../build/cylinder_fem_amr --steps 60 > run.log 2>&1 && echo "smoke ok" && grep -a 'host threads' run.log | head -1)
bash scripts/timing.sh build/cylinder_fem_amr "$OUT" "$REPS" 2>&1 | tee "$OUT/timing.out"
python3 scripts/efficiency_table.py "$OUT" U3 U4 U5 A1 A2 A3 | tee "$OUT/efficiency_table.txt"
tar czf "results_$GPU.tar.gz" --exclude='*.vtk' "$OUT"
echo "results: results_$GPU.tar.gz"
