#!/usr/bin/env bash
# Локально (Git Bash): створити GPU-сервер на Vultr, запустити bootstrap, стежити і ВИДАЛИТИ сервер,
# коли тренування завершилось (файл DONE) або вичерпано бюджет годин — щоб не палити кредити на простій.
#   bash gpu/vultr_run.sh vcg-a40-96c-480g-192vram lax 40
set -uo pipefail
PLAN=${1:?план}; REGION=${2:?регіон}; HOURS=${3:-40}
V=~/bin/vultr/vultr-cli.exe; KEY=02d687ef-545a-4bc9-b750-6a463468370b
LOG=/d/ai/outputs/gpu_run.log
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
HF_TOKEN=$(hf auth token 2>/dev/null | tail -1)
[ -n "$HF_TOKEN" ] || { say "немає HF-токена (hf auth login)"; exit 1; }

out=$($V instance create -p $PLAN -r $REGION --os 1743 --ssh-keys $KEY -l lewi-gpu --host lewi-gpu 2>&1 | grep -v "config file")
ID=$(echo "$out" | awk '/^ID/{print $2; exit}')
[ -n "$ID" ] || { say "створення не вдалося: $out"; exit 1; }
say "створено $ID ($PLAN, $REGION), бюджет ${HOURS} год"
START=$(date +%s)
destroy() { say "видаляю сервер $ID: $1"; $V instance delete $ID 2>&1 | grep -v "config file" | tee -a $LOG; exit 0; }

for i in $(seq 120); do
  info=$($V instance get $ID 2>/dev/null)
  IP=$(echo "$info" | awk '/^MAIN IP/{print $3}'); ST=$(echo "$info" | awk '/^SERVER STATUS/{print $3}')
  [ "$ST" = ok ] && [ -n "$IP" ] && [ "$IP" != 0.0.0.0 ] && break; sleep 20
done
say "IP $IP"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@$IP"
for i in $(seq 30); do $SSH true 2>/dev/null && break; sleep 20; done
scp -o StrictHostKeyChecking=no /d/ai/gpu/bootstrap.sh root@$IP:~/bootstrap.sh
$SSH "HF_TOKEN=$HF_TOKEN HOURS=$HOURS bash ~/bootstrap.sh" 2>&1 | tail -5 | tee -a $LOG
# якщо bootstrap поставив драйвер і перезавантажив сервер — запускаємо вдруге
if ! $SSH "tmux has-session -t lewi" 2>/dev/null; then
  sleep 120; for i in $(seq 30); do $SSH true 2>/dev/null && break; sleep 20; done
  $SSH "HF_TOKEN=$HF_TOKEN HOURS=$HOURS bash ~/bootstrap.sh" 2>&1 | tail -5 | tee -a $LOG
fi

last=""
while true; do
  el=$(( ($(date +%s) - START) / 3600 ))
  [ $el -ge $((HOURS + 1)) ] && destroy "вичерпано бюджет $HOURS год"
  $SSH "test -f ~/lewi/DONE" 2>/dev/null && { $SSH "cat ~/lewi/DONE; tail -3 ~/lewi/progress.log" | tee -a $LOG; destroy "тренування завершено"; }
  cur=$($SSH "tail -1 ~/lewi/progress.log" 2>/dev/null)
  [ -n "$cur" ] && [ "$cur" != "$last" ] && { say "$cur"; last=$cur; }
  if ! $SSH "tmux has-session -t lewi" 2>/dev/null; then
    say "tmux-сесія зникла — тренування впало? хвіст:"; $SSH "tail -20 ~/lewi/train.log" 2>&1 | tee -a $LOG
    destroy "процес тренування не працює"
  fi
  sleep 300
done
