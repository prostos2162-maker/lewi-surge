"""
Швидка causal depthwise Conv1D для Gated DeltaNet: замість cuDNN-згортки (на GPU повільна й обмежена пам'яттю)
робимо K зсунутих множень-додавань — XLA зливає їх в одне елементне ядро, пам'ять читається один раз.

y[t, c] = Σ_{j<K} w[j, 0, c] · x[t-(K-1)+j, c]   (точно те саме, що nnx.Conv(padding="CAUSAL", groups=C))
Підключення: import fast_conv; fast_conv.install() — ДО імпорту maxtext.
"""
import jax.numpy as jnp
from flax import nnx

_orig_call = nnx.Conv.__call__


def _fast_call(self, inputs, **kwargs):
    k = self.kernel_size if isinstance(self.kernel_size, (tuple, list)) else (self.kernel_size,)
    depthwise = self.feature_group_count == self.in_features == self.out_features
    if (len(k) != 1 or not depthwise or self.padding != "CAUSAL" or inputs.ndim != 3
            or self.bias is not None and self.bias.value is not None
            or getattr(self, "kernel_dilation", 1) not in (1, (1,), None)
            or getattr(self, "strides", 1) not in (1, (1,), None)):
        return _orig_call(self, inputs, **kwargs)
    ks = k[0]
    dtype = self.dtype or inputs.dtype
    x = inputs.astype(dtype)
    w = self.kernel.value.astype(dtype)  # [K, 1, C]
    t = x.shape[1]
    xp = jnp.pad(x, ((0, 0), (ks - 1, 0), (0, 0)))
    y = xp[:, 0:t, :] * w[0, 0]
    for j in range(1, ks):
        y = y + xp[:, j:j + t, :] * w[j, 0]
    return y


def install():
    nnx.Conv.__call__ = _fast_call
    print("[fast_conv] nnx.Conv(causal depthwise 1D) → shift-add", flush=True)
