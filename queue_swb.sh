#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
# SafeWall-style baseline (swb) and unbuffered diagnostic (xgbraw) on the four Theta windows. Restart-safe.
cd $ROOT/online
O=$OUT_DIR/online; R=$OUT_DIR; CE=$CENTILE_DIR
for y in 2017 2018 2019 2020; do
  if [ $y = 2017 ]; then t=theta; d=theta; else t=theta$y; d=theta$y; fi
  yy=t${y:2:2}
  for n in $(python3 -c "import json;print(' '.join(map(str,json.load(open('$R/$d/sizes.json'))['sizes'])))"); do
    for a in swb xgbraw; do
      f=$O/online_${a}_${yy}_$n.json
      [ -f $f ] && continue
      echo "START $a $t $n $(date +%T)"
      TAG=_$yy python3 -I online_sim.py $CE $t $n $a $O > logs/${a}_${yy}_$n.log 2>&1 || echo "FAILED $a $t $n"
      grep -E '"mean_bs"' logs/${a}_${yy}_$n.log | tr -d '\n'; echo
    done
  done
  echo "YEARDONE $y"
done
echo SWBDONE
