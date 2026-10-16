#!/bin/bash
# Bare-metal Vultr (Ubuntu 24.04): підготовка до токенізації етапу 1 для 7B-A1B.
# Викликається з ПК: scp tok_bootstrap.sh tok_stage1.py tokenizer.json root@IP:~/ && ssh root@IP "HF_TOKEN=... bash tok_bootstrap.sh"
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv tmux htop zstd nvme-cli > /dev/null
# другий диск (bare metal має 2×480 ГБ): змонтувати під /data, якщо є
if lsblk -dn -o NAME,MOUNTPOINT | awk '$2==""' | grep -q .; then
  D=$(lsblk -dn -o NAME,SIZE,MOUNTPOINT,TYPE | awk '$3=="" && $4=="disk"{print $1; exit}')
  if [ -n "$D" ] && ! blkid /dev/$D >/dev/null 2>&1; then mkfs.ext4 -q -F /dev/$D && mkdir -p /data && mount /dev/$D /data; fi
fi
mkdir -p /data/out /data/hf; chmod 777 /data
python3 -m venv ~/venv && ~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q tokenizers pyarrow huggingface_hub numpy zstandard
# gcloud CLI для вивантаження в бакет (авторизація — ADC-файл користувача, копіюється окремо)
if ! command -v gcloud >/dev/null; then
  curl -sSL https://sdk.cloud.google.com > /tmp/gc.sh && bash /tmp/gc.sh --disable-prompts --install-dir=/opt > /dev/null 2>&1
  ln -sf /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
fi
echo "export HF_TOKEN=${HF_TOKEN:-} HF_HOME=/data/hf" > ~/env.sh
nproc; free -g | head -2; df -h /data | tail -1; ~/venv/bin/python -c "import tokenizers, pyarrow; print('deps ok')"
echo BOOTSTRAP_DONE
