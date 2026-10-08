#!/usr/bin/env bash
# Нічний вартовий: статус Kaggle-ноутбука + ключові рядки живого лога. Кожен рядок stdout = подія.
K=${1:-steflett/moe-stage1-train}
cd /d/ai
prev_status=""
seen=0
while true; do
  python kaggle/refresh.py >/dev/null 2>&1 || echo "WARN: refresh токена не вдався"
  st=$(kaggle kernels status "$K" 2>&1 | sed -E 's/.*status "KernelWorkerStatus\.([A-Z_]+)".*/\1/' | tail -1)
  if [ "$st" != "$prev_status" ]; then
    echo "STATUS: $st ($(date +%H:%M))"
    prev_status=$st
  fi
  if [ "$st" = "RUNNING" ] || [ "$st" = "COMPLETE" ] || [ "$st" = "ERROR" ]; then
    kaggle kernels logs "$K" > out/kaggle_stage1_live.log 2>&1
    n=$(wc -l < out/kaggle_stage1_live.log)
    if [ "$n" -gt "$seen" ]; then
      tail -n +$((seen + 1)) out/kaggle_stage1_live.log \
        | grep -a -E "PROBE|-> ok|ОБРАНО|Беру збережений|ПРОДОВЖУЮ|старт з нуля|ТРЕНУВАННЯ|completed step|Traceback|Error|RESOURCE_EXHAUSTED|Дедлайн|Тренування завершилось|Чекпоінти у виводі|\[patch\]|exit [1-9]" \
        | grep -a -v "Config param" | cut -c1-260 | tail -n 12
      seen=$n
    fi
  fi
  case "$st" in
    COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCEL_REQUESTED) echo "FINAL: $st"; exit 0 ;;
  esac
  sleep 300
done
