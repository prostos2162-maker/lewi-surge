#!/usr/bin/env bash
# Стежить за 4 CPU-сесіями токенізації: прогрес кожні 5 хв, вихід коли всі завершились.
cd /d/ai
declare -A last
while true; do
  python kaggle/refresh.py >/dev/null 2>&1
  done_n=0
  for s in a b c d; do
    k=steflett/moe-tok-fineweb-edu-$s
    st=$(kaggle kernels status $k 2>&1 | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/' | tail -1)
    line=$(kaggle kernels logs $k 2>/dev/null | grep -a -oE '[0-9]{3}_00000\.parquet: [^"\\]*|ГОТОВО[^"\\]*|ПЕРЕВІРКА OK|Error[^"\\]*|Traceback' | tail -1)
    msg="$s: $st | $line"
    if [ "${last[$s]}" != "$msg" ]; then echo "$msg"; last[$s]=$msg; fi
    case "$st" in COMPLETE|ERROR|CANCEL*) done_n=$((done_n + 1)) ;; esac
  done
  [ $done_n -eq 4 ] && { echo "ALL TOKENIZATION FINISHED"; exit 0; }
  sleep 300
done
