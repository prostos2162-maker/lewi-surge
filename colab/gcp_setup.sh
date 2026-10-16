# Підготовка TPU VM (GCP v6e-1, Ubuntu з Python 3.10): uv + Python 3.12 + MaxText 0.2.5
set -x
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
cd ~ && rm -rf mt && uv venv mt --python 3.12 && source mt/bin/activate
uv pip install -q 'maxtext[tpu]==0.2.5' --resolution=lowest > pip.log 2>&1
uv pip install -q 'transformers==5.19.0' xprof tensorboard-plugin-profile pip > /dev/null 2>&1
python -c "import jax; print('JAX', jax.__version__, jax.devices())"
echo SETUP_DONE
