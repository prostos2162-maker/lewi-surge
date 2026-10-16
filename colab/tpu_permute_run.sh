#!/bin/bash
# Run the MoE permute kernel tests + microbenchmark on a Colab TPU session (default name: perm).
#   bash colab/tpu_permute_run.sh [session]
# Only measurements; no training is started.
S=${1:-perm}
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8  # Git Bash: do not rewrite /root paths; UTF-8 output
cd "$(dirname "$0")/.."
for f in modal/moe_permute.py modal/test_moe_permute.py colab/tpu_permute_bench.py; do
  colab upload -s $S $f /root/src/$(basename $f) >/dev/null || { echo "upload failed: $f"; exit 1; }
done
cat <<'PY' | colab exec -s $S
import subprocess, sys, os
env = dict(os.environ, TILES=os.environ.get("TILES", "64,128,256"), PYTHONUNBUFFERED="1")
r = subprocess.run([sys.executable, "/root/src/tpu_permute_bench.py"], env=env, capture_output=True, text=True)
print(r.stdout[-12000:]); print(r.stderr[-4000:]); print("RC", r.returncode)
PY
