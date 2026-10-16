# Сторож на самій TPU VM (7B): видаляє TPU після ALL_DONE або після 60 хв без тренувального процесу.
Z=${TPU_ZONE:-us-central1-b}; N=${TPU_NAME:-lewi7b}; idle=0
while true; do
  busy=$(pgrep -fc 'pre_train.train|gcp_lewi7b|kaggle|conv_wrapper|grow_7b')
  done_=$(grep -c ALL_DONE ~/go.log 2>/dev/null)
  if [ "$busy" = "0" ] || [ "${done_:-0}" != "0" ]; then idle=$((idle+1)); else idle=0; fi
  lim=12; [ "${done_:-0}" != "0" ] && lim=2
  echo "$(date +%H:%M) busy=$busy done=$done_ idle=${idle}x5хв"
  [ $idle -ge $lim ] && { echo "ВИДАЛЯЮ TPU $N"; gcloud compute tpus tpu-vm delete $N --zone=$Z --quiet --async; exit 0; }
  sleep 300
done
