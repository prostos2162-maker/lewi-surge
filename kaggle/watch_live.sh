#!/usr/bin/env bash
# Живий лог Kaggle-сесії: раз на хвилину забираємо лог, друкуємо нові важливі рядки; виходимо, коли сесія завершилась.
K=${1:-steflett/moe-stage1-train}
cd /d/ai
seen=0
i=0
while true; do
  [ $((i % 20)) -eq 0 ] && python kaggle/refresh.py >/dev/null 2>&1
  i=$((i + 1))
  kaggle kernels logs "$K" > out/kaggle_stage1_live.log 2>/dev/null
  n=$(wc -l < out/kaggle_stage1_live.log)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) out/kaggle_stage1_live.log | grep -a -v "Config param" \
      | grep -a -E "assets:|exit [0-9]|TpuDevice|PROBE|-> ok|хвіст|ОБРАНО|Беру збережений|ПРОДОВЖУЮ|старт з нуля|ТРЕНУВАННЯ|completed step|Traceback|Error|RESOURCE_EXHAUSTED|Дедлайн|завершилось|Чекпоінти|\[patch\]" \
      | cut -c1-260 | tail -n 15
    seen=$n
  fi
  st=$(kaggle kernels status "$K" 2>&1 | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/' | tail -1)
  case "$st" in RUNNING|QUEUED) ;; *) echo "FINAL STATUS: $st"; exit 0 ;; esac
  sleep 60
done
