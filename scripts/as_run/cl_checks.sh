#!/bin/bash
# C_L,max sensitivity checks on the default AMR configuration, started after the dt sweep
S=${SCRATCH:-/tmp/cfa_scratch}
while pgrep -f "/dt_solver3 " > /dev/null; do sleep 20; done
echo "dt sweep finished, starting C_L checks"
mkdir -p $S/clcheck/stab001 $S/clcheck/dt05
(cd $S/clcheck/stab001 && $S/cl_solver --steps 6000 --pressure-stab 0.01 --print-interval 1000 --stats-window 2 > run.log 2>&1; echo "stab 0.01 exit=$?") &
(cd $S/clcheck/dt05 && $S/cl_solver --steps 12000 --dt 0.0005 --amr-interval 20 --print-interval 2000 --stats-window 2 > run.log 2>&1; echo "dt 0.5ms exit=$?") &
wait
for d in stab001 dt05; do echo "== $d"; grep -A6 "Force statistics" $S/clcheck/$d/run.log; grep -E "Timing|non-finite|abort" $S/clcheck/$d/run.log | head -2; done
