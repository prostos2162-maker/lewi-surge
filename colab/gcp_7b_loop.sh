#!/bin/bash
# Цикл на ПК: тримає spot v6e-8 живим для тренування 7B. Створює TPU (перебирає зони), ставить MaxText, запускає
# gcp_lewi7b.sh; після витіснення/зникнення VM створює нову і продовжує з чекпойнта в бакеті. Зупиняється на STAGE_DONE.
#   bash colab/gcp_7b_loop.sh            # лог: outputs/gcp7b.log
#   HOURS=50 — жорсткий ліміт часу циклу; ZONES — список зон через пробіл
set -uo pipefail
LOG=/d/ai/outputs/gcp7b.log; N=${TPU_NAME:-lewi7b}; HOURS=${HOURS:-55}
ZONES=${ZONES:-"us-central1-b us-east1-d us-east5-a europe-west4-a asia-south1-c"}
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
START=$(date +%s)
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
tpu_zone() {  # list вимагає зону → перебираємо наші зони
  for z in $ZONES; do
    st=$(gcloud compute tpus tpu-vm describe $N --zone=$z --format="value(state)" 2>/dev/null | tr -d '\r')
    [ -n "$st" ] && { echo "$z $st"; return 0; }
  done
}
create() {
  for z in $ZONES; do
    say "створюю $N у $z (spot)"
    if gcloud compute tpus tpu-vm create $N --zone=$z --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot --quiet 2>&1 | tail -2 | tee -a $LOG | grep -q "ERROR"; then continue; fi
    sleep 20; st=$(tpu_zone | awk '{print $2}'); [ "$st" = READY ] && { echo $z; return 0; }
  done
  return 1
}
setup_and_run() {  # $1 zone
  local z=$1 S="gcloud compute tpus tpu-vm ssh $N --zone=$z --ssh-key-file=$HOME/.ssh/gcp_tpu --quiet"
  for i in $(seq 20); do $S --command="true" >/dev/null 2>&1 && break; sleep 20; done
  gcloud compute tpus tpu-vm scp /d/ai/colab/gcp_setup.sh /d/ai/colab/gcp_lewi7b.sh /d/ai/colab/vm_guard7b.sh /d/ai/configs/stage2.yml $HOME/.kaggle/credentials.json $N:~/ --zone=$z --ssh-key-file=$HOME/.ssh/gcp_tpu --quiet 2>&1 | tail -1
  $S --command="mv -f ~/credentials.json ~/kcred.json; sed -i 's/\r$//' ~/*.sh; bash ~/gcp_setup.sh > ~/setup.log 2>&1; tail -2 ~/setup.log" 2>&1 | tail -2 | tee -a $LOG
  $S --command="export TPU_ZONE=$z TPU_NAME=$N STAGE_TOKENS=${STAGE_TOKENS:-10000000000} PDB=${PDB:-16} REMAT=${REMAT:-minimal} VARIANT=${VARIANT:-k12}; nohup bash ~/gcp_lewi7b.sh > ~/go.log 2>&1 < /dev/null & disown; echo started" 2>&1 | tail -1 | tee -a $LOG
}
status() {  # $1 zone → останній рядок прогресу
  gcloud compute tpus tpu-vm ssh $N --zone=$1 --ssh-key-file=$HOME/.ssh/gcp_tpu --quiet --command="grep -E 'completed step' ~/lewi7b/train.log 2>/dev/null | tail -1 | cut -c1-160; grep -E 'STAGE_DONE|ALL_DONE|ВПАЛ|НЕМАЄ|ПОРОЖН' ~/go.log 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r'
}
while true; do
  el=$(( ($(date +%s) - START) / 3600 )); [ $el -ge $HOURS ] && { say "ліміт $HOURS год, зупиняю"; break; }
  info=$(tpu_zone); z=$(echo "$info" | awk '{print $1}'); st=$(echo "$info" | awk '{print $2}')
  if [ -z "$z" ] || [ "$st" = PREEMPTED ] || [ "$st" = TERMINATED ]; then
    [ -n "$z" ] && { say "TPU у стані $st → видаляю"; gcloud compute tpus tpu-vm delete $N --zone=$z --quiet 2>/dev/null; }
    gcloud storage ls gs://lewi-data/ckpt/7b_s1/STAGE_DONE >/dev/null 2>&1 && { say "етап завершено"; break; }
    z=$(create) || { say "немає ємності в жодній зоні, чекаю 10 хв"; sleep 600; continue; }
    say "TPU готовий у $z, налаштовую"; setup_and_run $z
  else
    s=$(status $z); say "[$z $st] $s"
    echo "$s" | grep -q STAGE_DONE && { say "STAGE_DONE → чекаю сторожа VM, виходжу"; gcloud storage cp /dev/null gs://lewi-data/ckpt/7b_s1/STAGE_DONE 2>/dev/null; break; }
    echo "$s" | grep -qE "ВПАЛ|НЕМАЄ|ПОРОЖН" && { say "ПОМИЛКА на VM, зупиняю цикл (TPU лишаю для розбору 60 хв — сторож видалить)"; break; }
  fi
  sleep 300
done
