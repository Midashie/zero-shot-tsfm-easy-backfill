#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
cd $ROOT/online
O=$OUT_DIR/online; R=$OUT_DIR; CE=$CENTILE_DIR
mkdir -p logs $O
for s in theta:4392 theta:3600 theta:5500; do
  t=${s%%:*}; n=${s##*:}
  if [ ! -f $R/rt_${t}_$n/contexts_c96.npz ]; then
    echo "BUILD $t $n $(date +%T)"; (cd ../rt && python3 -I build_rt.py $CE $t $n $R/rt_${t}_$n > $R/rt_${t}_$n.log 2>&1) || { echo "FAILED build $t $n"; continue; }
  fi
  if [ ! -f $O/online_validate_$n.json ] || ! grep -q '"trace": "'$t'"' $O/online_validate_$n.json; then
    echo "VALIDATE $t $n $(date +%T)"; python3 -I online_sim.py $CE $t $n validate $O > logs/validate_${t}_$n.log 2>&1 || { echo "FAILED validate $t $n"; continue; }
  fi
  bad=$(python3 -c "import json;print(json.load(open('$O/online_validate_$n.json'))['contexts_not_equal_to_build_rt'])")
  echo "VALIDATED $t $n mismatches=$bad"
  if [ "$bad" != "0" ]; then echo "DROPPED $t $n"; continue; fi
  for c in 96 192; do
    f=$O/online_bolt_th$c\_$n.json
    if [ -f $f ]; then continue; fi
    echo "START bolt ctx=$c $t $n $(date +%T)"
    CTX=$c TAG=_th$c python3 -I online_sim.py $CE $t $n bolt $O > logs/bolt_th${c}_${t}_$n.log 2>&1 || echo "FAILED bolt $c $t $n"
    grep -E '"mean_bs"|"seconds"' logs/bolt_th${c}_${t}_$n.log | tr -d '\n'; echo
  done
done
echo QUEUEDONE
