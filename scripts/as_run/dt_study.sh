#!/bin/bash
S=${SCRATCH:-/tmp/cfa_scratch}
cp $S/v10 $S/dt_solver
run_scheme() {
  scheme=$1
  for dt in 0.004 0.002 0.001 0.0005; do
    steps=$(python3 -c "print(round(6.0/$dt))")
    d=$S/dtstudy/${scheme}_$dt; mkdir -p $d; cd $d
    $S/dt_solver --steps $steps --dt $dt --amr-interval 0 --init-amr 2 --time-scheme $scheme \
       --print-interval 100000 --stats-window 2 > run.log 2>&1
    echo "$scheme dt=$dt exit=$?"
  done
}
run_scheme bdf2 &
run_scheme euler &
wait
