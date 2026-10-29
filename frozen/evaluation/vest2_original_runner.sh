#!/usr/bin/env bash
set -u
set -o pipefail

D="$1"
ROOT=/root/autodl-tmp/work2-algorithm-innovation
PY="$ROOT/sota_s5_dl85_official_R0/venv/bin/python"

export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "$$" > "$D/runner.pid"

{
  echo "PHASE=ENVIRONMENT_CAPTURE"
  date --iso-8601=seconds
  hostname
  "$PY" --version
  "$PY" -m pip show pydl8.5
  uptime
  free -h
  lscpu
  ps -eo pid,ppid,%cpu,%mem,stat,args \
    --sort=-%cpu | head -n 35
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used \
    --format=csv 2>/dev/null || true
} > "$D/environment.txt" 2>&1

echo "========== SOTA-S6 TRAIN100 THREE REPETITIONS =========="
echo "GPU_KEEPALIVE=UNCHANGED"
echo "WORK1=UNCHANGED"
echo "OUTPUT_DIR=$D"

"$PY" -B -u "$D/run.py" "$D" \
  2>&1 | tee "$D/run.log"

RC=${PIPESTATUS[0]}
echo "$RC" > "$D/exit_code.txt"

if [ "$RC" -eq 0 ] &&
   [ -s "$D/SUMMARY.json" ] &&
   [ ! -e "$D/FAIL_STOP.txt" ]; then
    echo "STATUS=PASS_S6_TRAIN100_EVIDENCE" > "$D/status.txt"
else
    echo "STATUS=FAIL_STOP_S6" > "$D/status.txt"
fi

(
  cd "$D" || exit 20
  sha256sum -c run_source.sha256 \
    > source_postflight.txt 2>&1
  if [ "$?" -ne 0 ]; then
    echo "STATUS=FAIL_STOP_SOURCE_DRIFT" > status.txt
  fi

  find . -type f \
    ! -name SHA256SUMS.txt -print0 |
    LC_ALL=C sort -z |
    xargs -0 -r sha256sum > SHA256SUMS.txt
)

echo "PHASE=PACKAGE_PAPER_EVIDENCE"

ARCH="$ROOT/$(basename "$D").evidence.tar.gz"

if [ -e "$ARCH" ]; then
  echo "STATUS=FAIL_STOP_ARCHIVE_COLLISION"
  exit 20
fi

if tar -czf "$ARCH" -C "$ROOT" "$(basename "$D")" &&
   [ -s "$ARCH" ]; then
    sha256sum "$ARCH" > "$ARCH.sha256"
    echo "ARCHIVE=$ARCH"
    cat "$ARCH.sha256"
else
    echo "STATUS=FAIL_STOP_PACKAGE" > "$D/status.txt"
fi

cat "$D/status.txt"
echo "OUTPUT_DIR=$D"
echo "========== SOTA-S6 FINISHED =========="
