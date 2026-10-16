#!/bin/bash
# Сторож кредитів Vultr (локально, Git Bash): видаляє ВСІ сервери, коли халявного кредиту лишається < $MIN
# або коли сторож пропрацював більше $HOURS годин. Баланс у Vultr від'ємний = кредит (-305 → $305).
#   bash vultr/credit_guard.sh [MIN=25] [HOURS=30]
MIN=${1:-25}; HOURS=${2:-30}
V=~/bin/vultr/vultr-cli.exe; LOG=/d/ai/outputs/vultr_guard.log
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
START=$(date +%s)
destroy_all() {
  for id in $($V instance list 2>/dev/null | awk '$1 ~ /^[0-9a-f]{8}-[0-9a-f-]{27}$/ {print $1}'); do
    say "видаляю $id: $1"; $V instance delete $id 2>&1 | grep -v "config file" | tee -a $LOG
  done
}
say "сторож: кредит < \$$MIN або > $HOURS год → видалення"
while true; do
  bal=$($V account info 2>/dev/null | awk 'NR==2{print $1}')
  credit=$(python -c "print(max(0.0, -float('${bal:-0}')))")
  n=$($V instance list 2>/dev/null | awk '$1 ~ /^[0-9a-f]{8}-[0-9a-f-]{27}$/' | wc -l)
  el=$(( ($(date +%s) - START) / 3600 ))
  say "кредит \$$credit, серверів $n, минуло ${el} год"
  if python -c "import sys; sys.exit(0 if $credit < $MIN else 1)"; then destroy_all "кредит нижче \$$MIN"; exit 0; fi
  [ $el -ge $HOURS ] && { destroy_all "ліміт $HOURS год"; exit 0; }
  [ "$n" = 0 ] && [ $el -ge 1 ] && { say "серверів немає, виходжу"; exit 0; }
  sleep 600
done
