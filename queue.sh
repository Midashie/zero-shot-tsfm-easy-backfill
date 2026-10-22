#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
cd $ROOT/online
O=$OUT_DIR/online
run() {  # arm ctx tag trace nodes
  f=$O/online_$1$3_$5.json
  if [ -f "$f" ]; then return; fi
  case $1 in moirai2) PY=$PY_M2;; ttm90) PY=$PY_TTM;; timesfm) PY=$PY_TF;; *) PY=python3;; esac
  echo "START $1 ctx=$2 $4 $5 $(date +%T)"
  CTX=$2 TAG=$3 $PY -I online_sim.py $CENTILE_DIR $4 $5 $1 $O > logs/$1$3_$4_$5.log 2>&1 || echo "FAILED $1 $4 $5"
  grep -E '"mean_bs"|"seconds"' logs/$1$3_$4_$5.log | tr -d '\n'; echo
}
mkdir -p logs
S="pm100:512 fdata:20000 fdata:23000"
for a in moirai2 ttm90; do for s in $S; do run $a 96 "" ${s%%:*} ${s##*:}; done; done
for c in 32 192; do for s in $S; do run bolt $c _c$c ${s%%:*} ${s##*:}; done; done
for a in chronos2 timesfm; do for s in $S; do run $a 96 "" ${s%%:*} ${s##*:}; done; done
echo QUEUEDONE
