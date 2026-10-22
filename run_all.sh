#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
cd $ROOT/online
O=$OUT_DIR/online
for a in tsafrir q90; do
  python3 -I online_sim.py $CENTILE_DIR pm100 512 $a $O
  python3 -I online_sim.py $CENTILE_DIR fdata 20000 $a $O
  python3 -I online_sim.py $CENTILE_DIR fdata 23000 $a $O
done
python3 -I online_sim.py $CENTILE_DIR pm100 512 bolt $O
python3 -I online_sim.py $CENTILE_DIR fdata 20000 bolt $O
python3 -I online_sim.py $CENTILE_DIR fdata 23000 bolt $O
echo ALLDONE
