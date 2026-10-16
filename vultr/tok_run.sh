#!/bin/bash
# Запуск усіх джерел етапу 1 паралельно на сервері (кожне джерело — свій процес, обмеження потоків на процес).
#   bash tok_run.sh            # усі джерела
#   P=6 bash tok_run.sh        # максимум 6 одночасно
source ~/env.sh
export RAYON_NUM_THREADS=${THREADS:-3} TOKENIZERS_PARALLELISM=true
mkdir -p /data/out /data/logs
~/venv/bin/python ~/tok_stage1.py --list
~/venv/bin/python ~/tok_stage1.py --list | awk 'NF==3' | awk '{print $1}' | \
  xargs -P ${P:-5} -I{} bash -c '~/venv/bin/python ~/tok_stage1.py {} > /data/logs/{}.log 2>&1; echo "done {} rc=$?"'
echo TOK_ALL_DONE
