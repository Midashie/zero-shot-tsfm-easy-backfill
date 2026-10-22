#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
O=$OUT_DIR/taska_full; mkdir -p $O; cd $ROOT/taska
for m in bolt chronos2; do echo "start $m $(date -u +%T)" >> $O/driver.log; python3 -I run_models.py $m $O > $O/$m.log 2>&1; echo "done $m $(date -u +%T) rc=$?" >> $O/driver.log; done
echo "start moirai2 $(date -u +%T)" >> $O/driver.log; $PY_M2 -I run_models.py moirai2 $O > $O/moirai2.log 2>&1; echo "done moirai2 $(date -u +%T) rc=$?" >> $O/driver.log
for m in ttm90 ttm180 ttm512; do echo "start $m $(date -u +%T)" >> $O/driver.log; $PY_TTM -I run_models.py $m $O > $O/$m.log 2>&1; echo "done $m $(date -u +%T) rc=$?" >> $O/driver.log; done
echo "start timesfm $(date -u +%T)" >> $O/driver.log; $PY_TF -I run_models.py timesfm $O > $O/timesfm.log 2>&1; echo "done timesfm $(date -u +%T) rc=$?" >> $O/driver.log
echo MODELS_DONE >> $O/driver.log
