#!/usr/bin/env bash
# Свіжий GPU-сервер (Ubuntu 22.04/24.04) → середовище MaxText + наші патчі → train_gpu.py у tmux.
#   HF_TOKEN=hf_... HOURS=40 bash bootstrap.sh
# CUDA (NVIDIA: A40, H100, ...) або ROCm (AMD MI300X) визначається автоматично.
set -euo pipefail
: "${HF_TOKEN:?потрібен HF_TOKEN (приватні дані й вивантаження)}"
W=~/lewi; mkdir -p $W/src
exec > >(tee -a $W/bootstrap.log) 2>&1

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then PLAT=cuda; nvidia-smi -L
elif command -v rocm-smi >/dev/null; then PLAT=rocm; rocm-smi --showproductname || true
else
  echo "немає драйвера GPU — ставлю NVIDIA (потім перезапусти bootstrap)"; sudo apt-get update -qq
  sudo apt-get install -y -qq ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot; exit 0
fi
echo "платформа: $PLAT"

sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev tmux git curl >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=~/.local/bin:$PATH
[ -d ~/mt ] || uv venv ~/mt --python 3.12
source ~/mt/bin/activate
# рецепт, що працював на Modal/Lightning: maxtext з мінімальними версіями, потім jax під свою платформу
uv pip install -q 'maxtext[tpu]==0.2.5' --resolution=lowest
uv pip uninstall -q libtpu libtpu-nightly scikit-learn 2>/dev/null || true
JV=$(python -c 'import jax; print(jax.__version__)')
if [ $PLAT = cuda ]; then uv pip install -q "jax[cuda12]==$JV"
else uv pip install -q "jax[rocm]==$JV" || uv pip install -q jax-rocm7-plugin jax-rocm7-pjrt  # ROCm: не перевірено
fi
uv pip install -q 'transformers==5.19.0' safetensors pyarrow 'huggingface_hub[cli]' \
  torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python -c "import jax; print('jax', jax.__version__, jax.devices())"
hf auth login --token "$HF_TOKEN" >/dev/null

# наш код: патчі (GPU-швидкість MoE/GDN), конвертер розмірів, конфіг
R=https://raw.githubusercontent.com/prostos2162-maker/lewi-surge/main
for f in modal/padded_gmm.py modal/fast_conv.py modal/gdn_fast.py kaggle/stage2/conv_wrapper.py configs/stage2.yml \
         gpu/train_gpu.py gpu/untie_head.py; do
  mkdir -p $W/src/$(dirname $f); curl -fsSL $R/$f -o $W/src/$f
  case $f in */*.py) cp $W/src/$f $W/src/ ;; esac
done

tmux kill-session -t lewi 2>/dev/null || true
tmux new -d -s lewi "source ~/mt/bin/activate && cd $W && HF_TOKEN=$HF_TOKEN HOURS=${HOURS:-40} SEQ=${SEQ:-8192} \
  python src/train_gpu.py 2>&1 | tee -a $W/train.log"
echo "BOOTSTRAP_OK — tmux attach -t lewi"
