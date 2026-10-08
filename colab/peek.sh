# Виводить хвіст логу з Colab-сесії: bash colab/peek.sh /content/smoke.log [N]
cd ~ && export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
timeout 120 colab exec -s ${S:-tpu1} </dev/null <<<"import subprocess;print(subprocess.run('tail -n ${2:-15} $1',shell=True,capture_output=True,text=True).stdout)" 2>&1 | tail -n ${2:-15}
rm -f ~/.config/colab-cli/colab.log
