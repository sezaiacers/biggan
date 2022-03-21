from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils import upsample, SpectralNormalization, downsample
import flax.linen.initializers as init


def get_kernel_initializer(initializer):
    if initializer == 'ortho':
        return init.orthogonal()
    elif initializer == 'N02':
        return init.normal(stddev=0.02)
    elif initializer in ['glorot', 'xavier']:
        return init.glorot_uniform()
    else:
        raise NotImplementedError(initializer)

class Attention(nn.Module):

    conv: nn.Module

    @nn.compact
    def __call__(self, inputs):
        ch = inputs.shape[-1]

        theta = self.conv(inputs, ch // 8, 1, use_bias=False)
        phi = nn.max_pool(
            self.conv(inputs, ch // 8, 1, use_bias=False),
            (2, 2), (2, 2))
        g = nn.max_pool(
            self.conv(inputs, ch // 2, 1, use_bias=False),
            (2, 2), (2, 2))

        probs = jax.nn.softmax(
            jnp.einsum('NHWC, NhwC -> NHWhw', theta, phi),
            (-1, -2))

        outputs = jnp.einsum('NHWhw, NhwC -> NHWC', probs, g)
        outputs = self.conv(outputs, ch, 1, use_bias=False)

        gamma = self.param('gamma', nn.initializers.zeros, (), jnp.float32)
        return outputs * gamma + inputs

class Generator(nn.Module):
    ch: int
    ch_mult: Sequence[int]
    attn_resolutions: Sequence[int]
    upsample_resolutions: Sequence[int]
    bottom_size: int
    shared_dim: int = 128
    shared_embeddings: bool = True
    channel_ratio: int = 4
    power_iterations: int = 1
    sn: bool = True
    hier: bool = True
    kernel_initializer: str = 'ortho'

    @nn.compact
    def __call__(self, inputs, labels, is_tranning):
        act = nn.relu
        kernel_init = get_kernel_initializer(self.kernel_initializer)

        def dense(inputs, ch, use_bias=True):
            fn = nn.Dense(ch, use_bias=use_bias, kernel_init=kernel_init)
            if not self.sn: return fn(inputs)

            fn = SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def conv(inputs, ch, kernel_size, use_bias=True):
            fn = nn.Conv(ch, (kernel_size, kernel_size),
                         strides=1, padding='SAME',
                         kernel_init=kernel_init,
                         use_bias=use_bias)
            if not self.sn: return fn(inputs)

            fn = SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def embedding(inputs, embed_dim):
            return nn.Dense(embed_dim, use_bias=False, kernel_init=kernel_init)(inputs)

        def bn(inputs, embeds):
            if self.shared_embeddings:
                scale = jnp.expand_dims(
                    dense(embeds, inputs.shape[-1], use_bias=False),
                    (1, 2))
                bias = jnp.expand_dims(
                    dense(embeds, inputs.shape[-1], use_bias=False),
                    (1, 2))
            else:
                scale = jnp.expand_dims(
                    embedding(embeds, inputs.shape[-1]),
                    (1, 2))
                bias = jnp.expand_dims(
                    embedding(embeds, inputs.shape[-1]),
                    (1, 2))

            norm = nn.BatchNorm(use_bias=False, use_scale=False, axis_name='batch')
            return norm(inputs, is_tranning) * (scale + 1.0) + bias

        def non_local_block(inputs):
            return Attention(conv)(inputs)

        def block(inputs, embeds, co, ch, upsample):
            hiddens = conv(act(bn(inputs, embeds)), ch, 1)
            hiddens = act(bn(hiddens, embeds))

            if inputs.shape[-1] != co:
                inputs = inputs[..., :co]

            if upsample:
                inputs, hiddens  = upsample(inputs), upsample(hiddens)

            hiddens = conv(hiddens, ch, 3)
            hiddens = conv(act(bn(hiddens, embeds)), ch, 3)
            hiddens = conv(act(bn(hiddens, embeds)), co, 1)
            return hiddens + inputs

        if self.shared_embeddings:
            embeds = embedding(labels, self.shared_dim)
        else:
            embeds = labels

        if self.hier:
            inputs = jnp.concatenate((inputs, embeds), -1)
            embeds = inputs

        for level, mult in enumerate(self.ch_mult):
            co, ch = self.ch * mult, self.ch * mult // self.channel_ratio

            if level == 0:
                inputs = jnp.reshape(
                    dense(inputs, co * (self.bottom_size ** 2)),
                    (-1, self.bottom_size, self.bottom_size, co))
            else:
                upsample_fn = upsample if level in self.upsample_resolutions else None
                inputs = block(inputs, embeds, co, ch, upsample_fn)

            if level in self.attn_resolutions:
                inputs = non_local_block(inputs)

        inputs = act(nn.BatchNorm(axis_name='batch')(inputs, is_tranning))
        return nn.tanh(conv(inputs, 3, 3))

class Discriminator(nn.Module):
    ch: int
    ch_mult: Sequence[int]
    attn_resolutions: Sequence[int]
    downsample_resolutions: Sequence[int]
    co: int = 1
    channel_ratio: int = 4
    power_iterations: int = 1
    kernel_initializer: str = 'ortho'

    @nn.compact
    def __call__(self, inputs, labels, is_tranning):
        act = nn.relu
        kernel_init = get_kernel_initializer(self.kernel_initializer)

        def dense(inputs, ch, use_bias=True):
            fn = nn.Dense(ch, use_bias=use_bias, kernel_init=kernel_init)
            fn = SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def conv(inputs, ch, kernel_size, use_bias=True):
            fn = nn.Conv(ch, (kernel_size, kernel_size),
                         strides=1, padding='SAME',
                         kernel_init=kernel_init,
                         use_bias=use_bias)
            fn = SpectralNormalization(fn, self.power_iterations)
            return fn(inputs, is_tranning)

        def embedding(inputs, embed_dim):
            return dense(inputs, embed_dim, use_bias=False)

        def non_local_block(inputs):
            return Attention(conv)(inputs)

        def block(inputs, co, ch, downsample):
            hiddens = conv(act(inputs), ch, 1)
            hiddens = conv(act(hiddens), ch, 3)
            hiddens = conv(act(hiddens), ch, 3)
            hiddens = act(hiddens)

            if downsample:
                hiddens = downsample(hiddens)

            hiddens = conv(hiddens, co, 1)

            if downsample:
                inputs = downsample(inputs)

            ci = inputs.shape[-1]
            if ci != co:
                inputs = jnp.concatenate(
                    [inputs, conv(inputs, co - ci, 1)],
                    -1)
            return hiddens + inputs

        for level, mult in enumerate(self.ch_mult):
            co, ch = self.ch * mult, self.ch * mult // self.channel_ratio
            if level == 0:
                hiddens = conv(inputs, co, 3)
            else:
                downsample_fn = downsample if level in self.downsample_resolutions else None
                hiddens = block(hiddens, co, ch, downsample_fn)

            if level in self.attn_resolutions:
                inputs = non_local_block(inputs)

        hiddens = jnp.sum(act(hiddens), (1, 2))
        outputs = dense(hiddens, self.co)

        projs = jnp.sum(embedding(labels, hiddens.shape[-1]) * hiddens, 1, keepdims=True)
        return outputs + projs

ch = 64
gen_attn_resolutions = (9, )
bottom_size = 4
upsample_resolutions = (2, 4, 6, 8, 10, 12)
ch_mult = (16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 2, 2, 1)

dis_ch_mult = (1, 2, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16)
downsample_resolutions = (1, 3, 6, 8, 10, 12)
dis_attn_resolutions = (5, )


model = Generator(ch, ch_mult, gen_attn_resolutions, upsample_resolutions, bottom_size)
#model = Discriminator(ch, dis_ch_mult, dis_attn_resolutions, downsample_resolutions)

key = jax.random.PRNGKey(0)

num_classes = 1000
inputs = jax.random.normal(key, (12, 10))
labels = jax.random.randint(key, (12, 1000), 0, num_classes)
#inputs = jax.random.normal(key, (12, 256, 256, 3))

variables = model.init(key, inputs, labels, True)
output = model.apply(variables, inputs, labels, True, mutable=['batch_stats', 'us'])
#output  = model.apply(variables, inputs, labels, False)
