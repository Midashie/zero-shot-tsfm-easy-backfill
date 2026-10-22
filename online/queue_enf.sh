#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
# Kill-and-requeue replay (ENFORCE=1) on the four Theta windows. Phase A: check + XGBoost arms. Phase B: Bolt-96 median. Phase C: Bolt-96 q0.8. Restart-safe.
cd $ROOT/online
O=$OUT_DIR/online; R=$OUT_DIR; CE=$CENTILE_DIR
tr_() { [ $1 = 2017 ] && echo theta || echo theta$1; }
sizes_() { python3 -c "import json;print(' '.join(map(str,json.load(open('$R/$(tr_ $1)/sizes.json'))['sizes'])))"; }
for y in 2017 2018 2019 2020; do
  t=$(tr_ $y); yy=${y:2:2}
  for n in $(sizes_ $y); do
    if [ ! -f $O/online_validate_e${yy}_$n.json ]; then
      ENFORCE=1 TAG=_e$yy python3 -I online_sim.py $CE $t $n validate $O > logs/validate_e${yy}_$n.log 2>&1 || { echo "FAILED validate $y $n"; exit 1; }
    fi
    old=$O/online_validate_$n.json; [ $y != 2017 ] && old=$O/online_validate_t${yy}_$n.json
    ok=$(python3 -c "
import json
a=json.load(open('$O/online_validate_e${yy}_$n.json')); b=json.load(open('$old'))
print(int(abs(a['mean_bs']-b['mean_bs'])<1e-9 and a['contexts_not_equal_to_build_rt']==0 and a['killed_attempts']==0))")
    echo "ENFCHECK $y $n ok=$ok"
    [ "$ok" = 1 ] || { echo "ENFFAIL $y $n"; exit 1; }
    for a in swb xgbraw; do
      [ -f $O/online_${a}_e${yy}_$n.json ] && continue
      ENFORCE=1 TAG=_e$yy python3 -I online_sim.py $CE $t $n $a $O > logs/${a}_e${yy}_$n.log 2>&1 || echo "FAILED $a $y $n"
      echo "DONE $a $y $n $(grep -E '"mean_bs"|"killed_jobs_pct"' logs/${a}_e${yy}_$n.log | tr -d '\n ')"
    done
  done
done
echo PHASEA
for y in 2017 2018 2019 2020; do
  t=$(tr_ $y); yy=${y:2:2}
  for n in $(sizes_ $y); do
    f=$O/online_bolt_e${yy}c96_$n.json; [ -f $f ] && continue
    oldtag=_t${yy}c96; [ $y = 2017 ] && oldtag=_th96
    [ -f $O/cache/bolt_e${yy}c96_${t}_$n.pkl ] || { [ -f $O/cache/bolt${oldtag}_${t}_$n.pkl ] && cp $O/cache/bolt${oldtag}_${t}_$n.pkl $O/cache/bolt_e${yy}c96_${t}_$n.pkl; }
    echo "START bolt median $y $n $(date +%T)"
    ENFORCE=1 CTX=96 TAG=_e${yy}c96 python3 -I online_sim.py $CE $t $n bolt $O > logs/bolt_e${yy}c96_${t}_$n.log 2>&1 || echo "FAILED bolt $y $n"
    echo "DONE bolt median $y $n $(grep -E '"mean_bs"|"killed_jobs_pct"|"seconds"' logs/bolt_e${yy}c96_${t}_$n.log | tr -d '\n ')"
  done
done
echo PHASEB
for y in 2017 2018 2019 2020; do
  t=$(tr_ $y); yy=${y:2:2}
  for n in $(sizes_ $y); do
    f=$O/online_bolt_e${yy}q80_$n.json; [ -f $f ] && continue
    echo "START bolt q80 $y $n $(date +%T)"
    ENFORCE=1 CTX=96 QIDX=7 TAG=_e${yy}q80 python3 -I online_sim.py $CE $t $n bolt $O > logs/bolt_e${yy}q80_${t}_$n.log 2>&1 || echo "FAILED bolt q80 $y $n"
    echo "DONE bolt q80 $y $n $(grep -E '"mean_bs"|"killed_jobs_pct"|"seconds"' logs/bolt_e${yy}q80_${t}_$n.log | tr -d '\n ')"
  done
done
echo ENFDONE
