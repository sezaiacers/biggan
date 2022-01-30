from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

import utils

from flax.linen.initializers import variance_scaling

ch = 64
attn_resolutions = ()
bottom_size = 4
upsample_resolutions = (2, 4, 6, 8, 10, 12)
ch_mult = (16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 2, 2, 1)

def default_kernel_init(scale, dtype=jnp.float32):
    scale=1e-10 if scale == 0 else scale
    return nn.initializers.variance_scaling(
        scale, mode='fan_avg', distribution='uniform',
        dtype=dtype)

class Generator(nn.Module):
    ch: int
    ch_mult: Sequence[int]
    attn_resolutions: Sequence[int]
    upsample_resolutions: Sequence[int]
    bottom_size: int
    channel_ratio: int = 4
    power_iterations: int = 1

    sn: bool = True

    @nn.compact
    def __call__(self, inputs, labels, is_tranning):
        act = nn.relu

        kernel_init = default_kernel_init(1.0)

        def dense(inputs, ch, use_bias=True):
            fn = nn.Dense(ch, use_bias=use_bias, kernel_init=kernel_init)
            if not self.sn: return fn(inputs)

            fn = utils.SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def conv(inputs, ch, kernel_size, use_bias=True):
            fn = nn.Conv(ch, (kernel_size, kernel_size),
                         strides=1, padding='SAME',
                         kernel_init=kernel_init)

            if not self.sn: return fn(inputs)

            fn = utils.SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def bn(inputs, embeds):
            norm = nn.BatchNorm(use_bias=False, use_scale=False, axis_name='batch')

            scale = jnp.expand_dims(
                dense(embeds, inputs.shape[-1], use_bias=False),
                (1, 2))

            bias = jnp.expand_dims(
                dense(embeds, inputs.shape[-1], use_bias=False),
                (1, 2))

            return norm(inputs, is_tranning) * scale + bias

        def block(inputs, embeds, co, ch, upsample=None):
            hiddens = conv(act(bn(inputs, embeds)), ch, 1)
            hiddens = act(bn(hiddens, embeds))

            inputs = inputs[..., :co]

            if upsample:
                inputs, hiddens  = upsample(inputs), upsample(hiddens)

            hiddens = conv(hiddens, ch, 3)
            hiddens = conv(act(bn(hiddens, embeds)), ch, 3)
            hiddens = conv(act(bn(hiddens, embeds)), co, 1)

            return hiddens + inputs


        inputs = jnp.concatenate((inputs, labels), -1)
        embeds = inputs

        oc = self.ch * self.ch_mult[0]

        inputs = dense(inputs, oc * (self.bottom_size ** 2))
        inputs = jnp.reshape(
            inputs,
            (-1, self.bottom_size, self.bottom_size, oc))

        for level, mult in enumerate(self.ch_mult):
            if level == 0: continue

            co, ch = self.ch * mult, self.ch * mult // self.channel_ratio
            upsample = utils.upsample if level in self.upsample_resolutions else None

            inputs = block(inputs, embeds, oc, ch, upsample)

            if level in self.attn_resolutions:
                pass

        inputs = act(nn.BatchNorm(axis_name='batch')(inputs, is_tranning))
        return nn.tanh(conv(inputs, 3, 3))


model = Generator(ch, ch_mult, attn_resolutions, upsample_resolutions, bottom_size)

key = jax.random.PRNGKey(0)

inputs = jax.random.normal(key, (12, 10))
labels = jax.random.normal(key, (12, 6))

variables = model.init(key, inputs, labels, True)
