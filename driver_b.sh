#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
O=$OUT_DIR/taskb; C=$O/contexts_c96.npz; cd $ROOT/taskb
run(){ echo "start $1 $(date -u +%T)" >> $O/driver_b.log; "${@:2}" > $O/$1.log 2>&1; echo "done $1 $(date -u +%T) rc=$?" >> $O/driver_b.log; }
run bolt python3 -I run_forecast_b.py bolt $C $O
run moirai2 $PY_M2 -I run_forecast_b.py moirai2 $C $O
run ttm90 $PY_TTM -I run_forecast_b.py ttm90 $C $O
run chronos2 python3 -I run_forecast_b.py chronos2 $C $O
run chronos2cov python3 -I run_forecast_b.py chronos2cov $C $O
echo FORECASTS_DONE >> $O/driver_b.log
