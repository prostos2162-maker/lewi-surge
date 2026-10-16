"""Вивантаження готових файлів токенізації в бакет через ADC (gcloud CLI без 'auth login' не працює).
   python gcs_up.py /data/out gs://lewi-data/stage1   # ллє *.bin/*.idx/*.json, пропускає вже наявні того ж розміру"""
import os, sys
from google.cloud import storage
src, dst = sys.argv[1], sys.argv[2]
bucket_name, prefix = dst[5:].split("/", 1)
client = storage.Client()
b = client.bucket(bucket_name)
for f in sorted(os.listdir(src)):
    if not f.endswith((".bin", ".idx", ".json")):
        continue
    p = os.path.join(src, f); size = os.path.getsize(p)
    blob = b.blob(f"{prefix}/{f}")
    if blob.exists() and (blob.reload() or blob.size == size):
        print("є", f); continue
    blob.chunk_size = 64 * 1024 * 1024
    blob.upload_from_filename(p, timeout=600)
    print("ok", f, f"{size/1e9:.2f} GB", flush=True)
