#!/bin/bash
# BDF2 dt sweep with a dt-independent SUPG tau (time scale fixed at 1 ms)
S=${SCRATCH:-/tmp/cfa_scratch}
run() {
  for dt in "$@"; do
    steps=$(python3 -c "print(round(6.0/$dt))")
    d=$S/dtstudy3/bdf2_$dt; mkdir -p $d; cd $d
    $S/dt_solver3 --steps $steps --dt $dt --amr-interval 0 --init-amr 2 --time-scheme bdf2 \
       --supg-tau-time 0.001 --print-interval 100000 --stats-window 2 > run.log 2>&1
    echo "bdf2 dt=$dt exit=$?"
  done
}
run 0.00025 &
run 0.004 0.002 0.001 0.0005 &
wait
