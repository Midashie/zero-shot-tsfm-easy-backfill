#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
W=$ROOT; cd $W/taskb; O=$OUT_DIR
for S in rt_pm100_512 rt_fdata_20000 rt_fdata_23000; do
  D=$O/$S; C=$D/contexts_c96.npz
  run(){ echo "start $S $1 $(date -u +%T)" >> $O/driver_rt.log; TAGS=win "${@:2}" >> $D/$1.log 2>&1; echo "done $S $1 $(date -u +%T) rc=$?" >> $O/driver_rt.log; }
  run bolt python3 -I run_forecast_b.py bolt $C $D
  run moirai2 $PY_M2 -I run_forecast_b.py moirai2 $C $D
  run ttm90 $PY_TTM -I run_forecast_b.py ttm90 $C $D
  run chronos2 python3 -I run_forecast_b.py chronos2 $C $D
done
echo FORECASTS_DONE >> $O/driver_rt.log
