#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
cd $ROOT/online
O=$OUT_DIR/online; CE=$CENTILE_DIR; mkdir -p logs
for s in pm100:512 pm100:420 pm100:640 fdata:16400 fdata:20000 fdata:23000 fdata:25000; do
  t=${s%%:*}; n=${s##*:}
  f=$O/online_bolt_q80_$n.json
  if [ -f $f ]; then continue; fi
  echo "START bolt q0.8 $t $n $(date +%T)"
  QIDX=7 TAG=_q80 python3 -I online_sim.py $CE $t $n bolt $O > logs/bolt_q80_${t}_$n.log 2>&1 || echo "FAILED bolt q80 $t $n"
  grep -E '"mean_bs"|"seconds"' logs/bolt_q80_${t}_$n.log | tr -d '\n'; echo
done
echo QUEUEDONE
