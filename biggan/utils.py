from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

def upsample(inputs, scale=2, method='nearest'):
    n, h, w, c = inputs.shape
    return jax.image.resize(inputs, (n, h * scale, w * scale, c), method=method)

def downsample(inputs):
    return nn.avg_pool(inputs, (2, 2), (2, 2))

class SpectralNormalization(nn.Module):
    module: nn.Module
    power_iterations: int = 1
    precision: Any = None
    params_name: str = 'params'
    kernel_name: str = 'kernel'

    @nn.compact
    def __call__(self, inputs, is_tranning):
        if not self.params_name in self.module.variables:
            self.module(inputs)

        params = self.module.variables[self.params_name]
        kernel = self.normalize(params[self.kernel_name], is_tranning)
        params = params.copy(add_or_replace={self.kernel_name: kernel})

        return self.module.clone(parent=None).apply(dict(params=params), inputs)

    def normalize(self, kernel, update_u):
        kernel_shape = kernel.shape
        kernel = kernel.reshape(-1, kernel_shape[-1])

        u = self.variable('us', 'u',
                          nn.initializers.normal(stddev=1.0),
                          (kernel.shape[0], 1))

        spectral_norm, new_u = compute_spectral_norm(
            kernel, u.value, self.power_iterations, self.precision)

        kernel = kernel / spectral_norm

        initializing = self.is_mutable_collection(self.params_name)
        if not initializing and update_u:
            u.value = new_u

        return kernel.reshape(*kernel_shape)

def compute_spectral_norm(w,
                          u,
                          power_iteration_rounds=1,
                          precision=None):

    for _ in range(power_iteration_rounds):
        v = l2_normalize(jnp.dot(w.T, u, precision=precision))
        u = l2_normalize(jnp.dot(w, v, precision=precision))

    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)

    spectral_norm = jnp.dot(
        jnp.dot(u.T, w, precision=precision),
        v,
        precision=precision)

    return spectral_norm, u

def l2_normalize(inputs, axis=None, epsilon=1e-12):
    return inputs * jax.lax.rsqrt(
        jnp.maximum(jnp.sum(jnp.square(inputs), axis=axis), epsilon))
