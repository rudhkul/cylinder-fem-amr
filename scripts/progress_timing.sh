#!/bin/bash
# Progress bars for scripts/timing.sh: scripts/progress_timing.sh [outdir] [reps]
OUT=${1:-runs/timing}; REPS=${2:-2}
bar() { local p=$1 w=30 f a b; f=$((p*w/100)); a=$(printf '%*s' $f ''); b=$(printf '%*s' $((w-f)) ''); printf '[%s%s] %3d%%' "${a// /#}" "${b// /.}" "$p"; }
done_n=0; total=$((6*REPS)); cur=""
for r in $(seq 1 $REPS); do for c in U3 U4 U5 A1 A2 A3; do
    n=${c}_r$r
    if [ -f $OUT/$n/done ]; then done_n=$((done_n+1)); w=$(grep -a -o 'Timing\] [0-9.]*' $OUT/$n/run.log | head -1 | cut -d' ' -f2); lastdone="$n ${w}s"
    elif [ -f $OUT/$n/run.log ]; then cur=$n; fi
done; done
printf 'campaign   '; bar $((done_n*100/total)); printf '  %d/%d runs done' $done_n $total; [ -n "${lastdone:-}" ] && printf '  (last: %s)' "$lastdone"; echo
if [ -n "$cur" ]; then
    line=$(grep -a '^\[Step' $OUT/$cur/run.log | tail -1)
    s=$(sed -E 's/^\[Step +([0-9]+).*/\1/' <<<"$line"); s=${s:-0}
    el=$(sed -E 's/.*Elems: +([0-9]+).*/\1/' <<<"$line")
    st=$(stat -c %W $OUT/$cur/run.log); dt=$(( $(date +%s)-st ))
    printf 'current    %-7s ' $cur; bar $((s*100/8000)); printf '  step %d/8000  %s elems  %ds elapsed\n' $s "${el:-0}" $dt
fi
