import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

from hyperparameters import hps


class ConvBlock(layers.Layer):
    def __init__(self, filters, activation=layers.LeakyReLU(0.2),
                 kernel_size=(5, 5), strides=(2, 2),
                 padding="same", use_bias=True,
                 use_bn=False, use_dropout=False,
                 drop_value=0.3):

        super(ConvBlock, self).__init__()

        self._conv = layers.Conv2D(
            filters, kernel_size, strides=strides,
            padding=padding, use_bias=use_bias
        )

        if use_bn:
            self._bn = layers.BatchNormalization()
        else:
            self._bn = None

        if use_dropout:
            self._dropout = layers.Dropout(drop_value)
        else:
            self._dropout = None

        self._activation = activation

    def call(self, inputs):
        x = self._conv(inputs)
        if self._bn is not None:
            x = self._bn(x)
        x = self._activation(x)
        if self._dropout is not None:
            x = self._dropout(x)

        return x


class UpsampleBlock(layers.Layer):
    def __init__(self, filters, activation=layers.LeakyReLU(0.2),
                 kernel_size=(3, 3), strides=(1, 1),
                 up_size=(2, 2), padding="same",
                 use_bias=True, use_bn=False,
                 use_dropout=False, drop_value=0.3):

        super(UpsampleBlock, self).__init__()

        self._upsample = layers.UpSampling2D(up_size)
        self._conv = layers.Conv2D(
            filters, kernel_size, strides=strides,
            padding=padding, use_bias=use_bias
        )
        self._activation = activation

        if use_bn:
            self._bn = layers.BatchNormalization()
        else:
            self._bn = None

        if use_dropout:
            self._dropout = layers.Dropout(drop_value)
        else:
            self._dropout = None

    def call(self, inputs):
        x = self._upsample(inputs)
        x = self._conv(x)
        if self._bn is not None:
            x = self._bn(x)
        x = self._activation(x)
        if self._dropout is not None:
            x = self._dropout(x)

        return x


class Generator(models.Model):
    def __init__(self, i):

        super(Generator, self).__init__(name="generator_"+str(i))

        self._dense = layers.Dense(4 * 4 * 256, use_bias=False)
        self._bn = layers.BatchNormalization()
        self._activation = layers.LeakyReLU(0.2)
        self._reshape = layers.Reshape((4, 4, 256))

        self._upsample1 = UpsampleBlock(128, layers.LeakyReLU(0.2),
                                        strides=(1, 1), use_bias=False,
                                        use_bn=True, padding="same",
                                        use_dropout=False)

        self._upsample2 = UpsampleBlock(64, layers.LeakyReLU(0.2),
                                        strides=(1, 1), use_bias=False,
                                        use_bn=True, padding="same",
                                        use_dropout=False)

        self._upsample3 = UpsampleBlock(1, layers.Activation("tanh"),
                                        strides=(1, 1), use_bias=False,
                                        use_bn=True)

    def call(self, inputs):
        x = self._dense(inputs)
        x = self._bn(x)
        x = self._activation(x)
        x = self._reshape(x)
        x = self._upsample1(x)
        x = self._upsample2(x)
        x = self._upsample3(x)

        return x


class Classiminator(models.Model):
    def __init__(self):

        super(Classiminator, self).__init__(name="classiminator")

        self._conv_block1 = ConvBlock(64)
        self._conv_block2 = ConvBlock(128, use_dropout=True)
        self._conv_block3 = ConvBlock(256, use_dropout=True)
        self._conv_block4 = ConvBlock(512)

        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(0.2)
        self._activation = layers.LeakyReLU(0.2)
        self._disc = layers.Dense(1)
        self._classes = layers.Dense(hps.num_classes)

    def call(self, inputs):
        x = self._conv_block1(inputs)
        x = self._conv_block2(x)
        x = self._conv_block3(x)
        x = self._conv_block4(x)
        x = self._flatten(x)
        x = self._dropout(x)

        classes = self._classes(x)
        x = self._activation(classes)
        disc = self._disc(x)

        return disc, classes


class WGCN_GP(models.Model):
    def __init__(self, from_ckpt=False):

        super(WGCN_GP, self).__init__(name='wgcn_gp')

        self.classiminator = Classiminator()
        if from_ckpt:
            self.classiminator.load_weights(hps.savedir+'classiminator'+".h5")

        self.num_gens = hps.num_gens
        self.generators = []
        for i in range(self.num_gens):
            self.generators.append(Generator(i))
        if from_ckpt:
            for i in range(self.num_gens):
                self.generators[i].load_weights(hps.savedir+"gen{}".format(i)+".h5")

        self.latent_dim = hps.noise_dim
        self.c_steps = hps.disc_iters_per_gen_iter
        self.gp_weight = hps.gp_weight
        self.batch_size = hps.batch_size
        self.num_classes = hps.num_classes

    def compile(self, c_optimizer, g_optimizers, d_loss_fn, c_loss_fn, g_loss_fn):
        super(WGCN_GP, self).compile()
        self.c_opt = c_optimizer
        self.g_opts = g_optimizers
        self.d_loss_fn = d_loss_fn
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, real_images, fake_images):
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred, _ = self.classiminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def gen_train_step(self, gen_num):
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            generated_images = self.generators[gen_num](random_latent_vectors, training=True)
            gen_img_logits, _ = self.classiminator(generated_images, training=True)
            g_cost = self.g_loss_fn(gen_img_logits)
        gen_gradient = tape.gradient(g_cost, self.generators[gen_num].trainable_variables)
        self.g_opts[gen_num].apply_gradients(
            zip(gen_gradient, self.generators[gen_num].trainable_variables)
        )

        return g_cost

    def train_step(self, real_data):
        real_images = real_data[0]
        real_classes = real_data[1]
        fake_classes = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        fake_classes[:, -1] = 1.0
        return_metrics = {}
        for i in range(self.c_steps):
            random_latent_vectors = tf.random.normal(
                shape=(self.batch_size, self.latent_dim)
            )

            rvs = tf.split(random_latent_vectors, self.num_gens)

            with tf.GradientTape() as tape:
                fake_images = tf.nest.map_structure(
                    lambda gen, rv: gen(rv, training=True),
                    self.generators, rvs
                )
                fake_images = tf.concat(fake_images, axis=0)

                fake_d, fake_c = self.classiminator(fake_images, training=True)
                real_d, real_c = self.classiminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_d, fake_img=fake_d)
                c_cost = self.c_loss_fn(real_img=real_c, fake_img=fake_c,
                                        real_true=real_classes,
                                        fake_true=fake_classes)
                gp = self.gradient_penalty(real_images, fake_images)

                total_cost = c_cost + d_cost + gp * self.gp_weight

            c_grads = tape.gradient(total_cost, self.classiminator.trainable_variables)
            self.c_opt.apply_gradients(
                zip(c_grads, self.classiminator.trainable_variables)
            )
            return_metrics["c_loss"] = total_cost

        """ for i in range(self.num_gens):
                g_cost = self.gen_train_step(i)
                return_metrics["g"+str(i)+"_loss"] = g_cost"""

        rvs = tf.random.normal(shape=(self.num_gens*self.batch_size, self.latent_dim))
        rvs = tf.split(rvs, self.num_gens, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            fake_images = tf.nest.map_structure(
                lambda gen, rv: gen(rv, training=True),
                self.generators, rvs
            )
            fake_images = tf.concat(fake_images, axis=0)

            gen_d, gen_c = self.classiminator(fake_images, training=True)
            gen_d = tf.split(gen_d, self.num_gens, axis=0)

            gen_losses = tf.nest.map_structure(
                lambda discs: self.g_loss_fn(discs, gen_c),
                gen_d
            )

        for i in range(self.num_gens):
            g_i_grad = tape.gradient(gen_losses[i], self.generators[i].trainable_variables)
            self.g_opts[i].apply_gradients(
                zip(g_i_grad, self.generators[i].trainable_variables)
            )
            return_metrics["g" + str(i) + "_loss"] = gen_losses[i]

        return return_metrics
    