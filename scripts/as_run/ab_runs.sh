#!/bin/bash
S=${SCRATCH:-/tmp/cfa_scratch}
(cd $S/ab/convective && $S/ab_solver --steps 6000 --convection-form convective --print-interval 1000 --stats-window 2 > run.log 2>&1; echo "convective exit=$?") &
(cd $S/ab/fine && $S/ab_solver --steps 6000 --amr-tol-vort 0.05 --amr-tol-interp 0.005 --max-level 5 --max-elems 400000 --print-interval 500 --stats-window 2 > run.log 2>&1; echo "fine exit=$?") &
wait
for d in convective fine; do echo "== $d"; grep -A6 "Force statistics" $S/ab/$d/run.log; grep -E "Timing|non-finite|abort" $S/ab/$d/run.log | head -2; echo "peak elems: $(grep 'mesh:' $S/ab/$d/run.log | awk '{print $4}' | sort -n | tail -1)"; done
