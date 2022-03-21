import jax
import jax.numpy as jnp


def subset(inputs, index, step_size):
    start_indices = (index * step_size, ) + (0, ) * len(inputs.shape[1:])
    slice_sizes = (step_size, ) + inputs.shape[1:]
    return jax.lax.dynamic_slice(inputs, start_indices, slice_sizes)


def make_generator_update_fn(discriminator_fn,
                                 generator_fn,
                                 criterion_fn,
                                 regularizer_fn,
                                 lr_fn,
                                 accum_steps):
    def update_fn(opt, dis_variables, gen_variables, step, noise, gen_labels):

        def loss_fn(params, gen_variables, noise, gen_labels):
            gen_images, new_gen_variables = generator_fn(
                {'params': params, 'us': gen_variables['us'], gen_variables['batch_stats']},
                noise, gen_labels, True, mutable=['us', 'batch_stats'])
            values = discriminator_fn(dis_variables, gen_images, gen_labels, False)
            loss = criterion_fn(values)
            return loss, new_gen_variables

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        def loss_and_grad_fn(params, gen_variables, noise, gen_labels):
            (loss, (new_gen_variables, )), grad = grad_fn(params, gen_variables, noise, gen_labels)
            return loss, new_gen_variables, grad

        if accum_steps and accum_steps > 1:
            step_size = noise.shape[0] // accum_steps
            loss, new_gen_variables, grad = loss_and_grad_fn(
                opt.params,
                gen_variables,
                subset(0, noise, step_size),
                subset(0, gen_labels, step_size))

            def acc_grad_and_loss(i, aux):
                prev_loss, prev_new_dis_variables, prev_grad = aux
                loss, new_dis_variables, grad = loss_and_grad_fn(
                    opt.params,
                    prev_new_dis_variables,
                    subset(i, noise, step_size),
                    subset(i, gen_labels, step_size))

                loss = loss + prev_loss
                grad = jax.tree_multimap(lambda x, y: x + y, grad, prev_grad)
                return loss, new_dis_variables, grad

            loss, new_gen_variables, grad = jax.lax.fori_loop(1, accum_steps,
                                                              (loss, new_gen_variables, grad))
            loss, grad = jax.tree_map(lambda x: x / accum_steps, (loss, grad))
        else:
            loss, new_gen_variables, grad  = loss_and_grad_fn(opt.params,
                                                            gen_variables,
                                                            noise,
        loss = jax.lax.pmean(loss, 'batch')
        grad = jax.lax.pmean(grad, 'batch')

        opt = opt.apply_gradient(grad)
        return opt, loss, new_gen_variables

    return jax.pmap(update_fn, axis_name='batch')

def make_discriminator_update_fn(discriminator_fn,
                                 generator_fn,
                                 criterion_fn,
                                 regularizer_fn,
                                 lr_fn,
                                 accum_steps):

    def update_fn(opt, dis_variables, gen_variables, step, real_images, real_labels, noise, gen_labels):
        def loss_fn(params, dis_variables, real_images, real_labels, noise, gen_labels):
            gen_images = generator_fn(gen_variables, noise, gen_labels, False)

            images = jnp.concatenate([real_images, gen_images], 0)
            labels = jnp.concatenate([real_labels, gen_labels], 0)

            values, new_dis_variables = discriminator_fn(
                {'params': params, 'us': dis_variables['us']},
                images, labels, True,
                mutable=['us'])

            real_values, gen_values = jnp.split(values, [real_images.shape[0]], 0)
            real_loss, gen_loss = criterion_fn(real_values, gen_values)
            loss = real_loss + gen_loss
            return loss, (real_loss, gen_loss, new_dis_variables)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        def loss_and_grad_fn(params, dis_variables, real_images, real_labels, noise, gen_labels):
            (loss, (real_loss, gen_loss, new_dis_variables)), grad = grad_fn(
                params, real_images, real_labels, noise, gen_labels)
            return loss, real_loss, gen_loss, new_dis_variables, grad

        if accum_steps and accum_steps > 1:
            dis_step_size = real_images.shape[0] // accum_steps
            gen_step_size = noise.shape[0] // accum_steps

            loss, real_loss, gen_loss, new_dis_variables, grad = loss_and_grad_fn(
                opt.target,
                dis_variables,
                subset(real_images, 0, dis_step_size),
                subset(real_labels, 0, dis_step_size),
                subset(noise, 0, gen_step_size),
                subset(gen_labels, 0, gen_step_size))

            def acc_grad_and_loss(i, aux):
                prev_loss, prev_real_loss, prev_gen_loss, prev_new_dis_variables, prev_grad = aux
                loss, real_loss, gen_loss, new_dis_variables, grad = loss_and_grad_fn(
                    opt.target,
                    prev_new_dis_variables,
                    subset(real_images, i, dis_step_size),
                    subset(real_labels, i, dis_step_size),
                    subset(noise, i, gen_step_size),
                    subset(gen_labels, i, gen_step_size))

                loss, real_loss, gen_loss, = loss + prev_loss, real_loss + prev_real_loss, gen_loss + prev_gen_loss
                grad = jax.tree_multimap(lambda x, y: x + y, grad, prev_grad)
                return loss, real_loss, gen_loss, new_dis_variables, grad

            loss, real_loss, gen_loss, new_dis_variables, grad = jax.lax.fori_loop(1, accum_steps,
                                                                                   (loss, real_loss, gen_loss, new_dis_variables, grad))
            loss, real_loss, gen_loss, grad = jax.tree_map(lambda x: x / accum_steps, (loss, real_loss, gen_loss, grad))
        else:
            loss, real_loss, gen_loss, new_dis_variables, grad = loss_and_grad_fn(
                opt.target, dis_variables, real_images, real_labels, noise, gen_labels)

        loss = jax.lax.pmean(loss, 'batch')
        real_loss = jax.lax.pmean(real_loss, 'batch')
        gen_loss = jax.lax.pmean(gen_loss, 'batch')
        grad = jax.lax.pmean(grad, 'batch')

        opt = opt.apply_gradient(grad)
        return opt, new_dis_variables, loss, real_loss, gen_loss
    return jax.pmap(update_fn, axis_name='batch')



def train(config, workdir):
    tf.io.gfile.makedirs(workdir)

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng, 2)

    train_ds, valid_ds = input_pipeline.get_datasets(config)

    discriminator_fn, discriminator_variables = initialize_discriminator(config, init_rng_discriminator)
    generator_fn, generator_variables = initialize_generator(config, init_rng_generator)
