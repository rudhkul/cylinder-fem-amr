#!/bin/bash
# Progress bar for the 2D-3 dt study: scripts/progress.sh [outdir]   (default runs/d3dt)
OUT=${1:-runs/d3dt}
declare -A STEPS=( [A2_dt025]=32000 [U4_dt05]=16000 [U4_dt025]=32000 )
bar() { local p=$1 w=30 f a b; f=$((p*w/100)); a=$(printf '%*s' $f ''); b=$(printf '%*s' $((w-f)) ''); printf '[%s%s] %3d%%' "${a// /#}" "${b// /.}" "$p"; }
for n in A2_dt025 U4_dt05 U4_dt025; do
    tot=${STEPS[$n]}; log=$OUT/$n/run.log
    if [ -f $OUT/$n/done ]; then printf '%-10s ' $n; bar 100; echo "  done"; continue; fi
    if [ ! -f $log ]; then printf '%-10s ' $n; bar 0; echo "  queued"; continue; fi
    line=$(grep -a '^\[Step' $log | tail -1)
    s=$(sed -E 's/^\[Step +([0-9]+).*/\1/' <<<"$line"); s=${s:-0}
    t=$(sed -E 's/.*t = ([0-9.]+) s.*/\1/' <<<"$line")
    el=$(sed -E 's/.*Elems: +([0-9]+).*/\1/' <<<"$line")
    st=$(stat -c %W $log); now=$(date +%s); dt=$((now-st))
    eta=$(( s>0 ? dt*(tot-s)/s : 0 ))
    printf '%-10s ' $n; bar $((s*100/tot)); printf '  step %d/%d  t=%s s  %s elems  elapsed %dm  ETA ~%dm (linear)\n' $s $tot "$t" "$el" $((dt/60)) $((eta/60))
done
