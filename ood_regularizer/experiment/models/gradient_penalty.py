import tensorflow as tf
import tfsnippet as spt


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def get_gradient_penalty(input_x, sample_x, D, batch_size, x_shape, algorithm='interpolate',
                         gradient_penalty_weight=2.0, gradient_penalty_index=6.0):
    x = tf.reshape(input_x, (-1,) + x_shape)
    x_ = tf.reshape(sample_x, (-1,) + x_shape)
    alpha = tf.random_uniform(tf.concat([[batch_size], [1] * len(x_shape)], axis=0), minval=0, maxval=1.0)
    differences = x - x_
    interpolates = x_ + alpha * differences

    gradient_penalty = 0.0

    if algorithm == 'interpolate':
        D_interpolates = D(interpolates)
        gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
        gradient_penalty = tf.sqrt(tf.reduce_sum(gradient_penalty, tf.range(-len(x_shape), 0)))
        gradient_penalty = gradient_penalty ** 2
        gradient_penalty = tf.pow(gradient_penalty, gradient_penalty_index / 2.0)
        gradient_penalty = tf.reduce_mean(gradient_penalty) * gradient_penalty_weight

    if algorithm == 'interpolate-gp':
        D_interpolates = D(interpolates)
        gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
        gradient_penalty = tf.sqrt(tf.reduce_sum(gradient_penalty, tf.range(-len(x_shape), 0))) - 1.0
        gradient_penalty = gradient_penalty ** 2
        gradient_penalty = tf.pow(gradient_penalty, gradient_penalty_index / 2.0)
        gradient_penalty = tf.reduce_mean(gradient_penalty) * gradient_penalty_weight

    if algorithm == 'both':
        # Sample from fake and real
        energy_real = D(x)
        energy_fake = D(x_)
        gradient_penalty_real = tf.square(tf.gradients(energy_real, [x.tensor if hasattr(x, 'tensor') else x])[0])
        gradient_penalty_real = tf.reduce_sum(gradient_penalty_real, tf.range(-len(x_shape), 0))
        gradient_penalty_real = tf.pow(gradient_penalty_real, gradient_penalty_index / 2.0)

        gradient_penalty_fake = tf.square(tf.gradients(energy_fake, [x_.tensor if hasattr(x_, 'tensor') else x_])[0])
        gradient_penalty_fake = tf.reduce_sum(gradient_penalty_fake, tf.range(-len(x_shape), 0))
        gradient_penalty_fake = tf.pow(gradient_penalty_fake, gradient_penalty_index / 2.0)

        gradient_penalty = (tf.reduce_mean(gradient_penalty_fake) + tf.reduce_mean(gradient_penalty_real)) \
                           * gradient_penalty_weight / 2.0
    return gradient_penalty
