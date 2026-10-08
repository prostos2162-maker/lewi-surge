"""Аварійна зупинка студії + діагностика шляхів. python lightning/stop.py"""
from lightning_sdk import Studio

s = Studio(name="moe-gpu-smoke", teamspace="deploy-model-project", user="prostos2162-maker")
print("status:", s.status)
if str(s.status).lower().endswith("running"):
    print(s.run('echo HOME=$HOME; pwd; ls ~; ls ~/moe; find / -name run.sh -path "*moe*" 2>/dev/null | head -3'))
    s.stop()
print("status after:", s.status)
