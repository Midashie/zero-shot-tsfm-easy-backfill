#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CENTILE_DIR="${CENTILE_DIR:-$ROOT/centile}"
PY_M2="${PY_M2:-python3}"; PY_TTM="${PY_TTM:-python3}"; PY_TF="${PY_TF:-python3}"
export OUT_DIR DATA_DIR CENTILE_DIR
# usage: driver.sh STAGES...   each stage is  base:SEED  or  causal:SEED
# Everything runs one after another on the 2 CPU cores. Results are copied to $OUT_DIR/cr.
export HOME=${HOME:-/root}
REPO=$CENTILE_DIR
W=$ROOT/cr
RES=$W/runs; mkdir -p $RES $OUT_DIR/cr
export ARM=HSTU LMAX=96 D=256 H=8 L=4 EP=30 MIX=6 STRIDE=24 QS=0.5,0.6
for st in "$@"; do
  kind=${st%%:*}; seed=${st##*:}; out=$RES/${kind}_s$seed; mkdir -p $out
  echo "=== $(date -u +%T) start $kind seed $seed" | tee -a $W/driver.log
  if [ "$kind" = base ]; then
    (cd $REPO && SEED=$seed OUT=$out python3 pretrain/run_backfill_model.py) > $out/train.log 2>&1
  else
    (cd $W && REPO=$REPO SEED=$seed OUT=$out SHIFT=$([ "$seed" = 0 ] && echo 1 || echo 0) python3 -I run_backfill_model_causal.py) > $out/train.log 2>&1
  fi
  echo "=== $(date -u +%T) trained $kind seed $seed: $(tail -n 1 $out/train.log)" | tee -a $W/driver.log
  for N in 20000 23000; do
    (cd $REPO && NODES=$N HISTQ=0.9 MODEL_EST=$out/hstu_q50.npz python3 pretrain/run_backfill_replay.py) > $out/replay_$N.log 2>&1
    echo "    replay $N: $(grep '^model' $out/replay_$N.log)" | tee -a $W/driver.log
  done
  mkdir -p $OUT_DIR/cr/${kind}_s$seed && cp $out/*.log $out/*.json $out/hstu_q50.npz $OUT_DIR/cr/${kind}_s$seed/ 2>/dev/null
done
echo "=== $(date -u +%T) DRIVER_DONE $*" | tee -a $W/driver.log
