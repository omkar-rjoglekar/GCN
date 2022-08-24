import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

from hyperparameters import hps


class ConvBlock(layers.Layer):
    def __init__(self, filters, activation=layers.LeakyReLU(0.2),
                 kernel_size=(5, 5), strides=(2, 2),
                 padding="same", use_bias=True,
                 use_ln=False, use_dropout=False,
                 use_bn=False, drop_value=0.3):

        super(ConvBlock, self).__init__()

        self._conv = layers.Conv2D(
            filters, kernel_size, strides=strides,
            padding=padding, use_bias=use_bias
        )

        if use_ln:
            self._bn = layers.LayerNormalization()
        elif use_bn:
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
    def __init__(self, i, use_cropping=False):

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

        self.use_cropping = use_cropping
        if self.use_cropping:
            self.crop = layers.Cropping2D((2, 2))

    def call(self, inputs):
        x = self._dense(inputs)
        x = self._bn(x)
        x = self._activation(x)
        x = self._reshape(x)
        x = self._upsample1(x)
        x = self._upsample2(x)
        x = self._upsample3(x)

        if self.use_cropping:
            x = self.crop(x)

        return x


class Discriminator(models.Model):
    def __init__(self):

        super(Discriminator, self).__init__(name="discriminator")

        self._conv_block1 = ConvBlock(64)
        self._conv_block2 = ConvBlock(128, use_ln=True, use_dropout=True)
        self._conv_block3 = ConvBlock(256, use_ln=True, use_dropout=True)
        self._conv_block4 = ConvBlock(512)

        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(0.2)
        self._disc = layers.Dense(1)

    def call(self, inputs):
        x = self._conv_block1(inputs)
        x = self._conv_block2(x)
        x = self._conv_block3(x)
        x = self._conv_block4(x)
        x = self._flatten(x)
        x = self._dropout(x)

        disc = self._disc(x)

        return disc

class MLP(layers.Layer):
    def __init__(self, layer_sizes=hps.classifier_mlp+[hps.num_classes]):
        super(MLP, self).__init__(name='MLP')

        self._layers = []
        for i in range(len(layer_sizes) - 1):
            self._layers.append(layers.Dense(layer_sizes[i], activation='relu'))
        self._layers.append(layers.Dense(layer_sizes[-1]))

    def call(self, inputs):
        x = inputs
        for i in range(len(self._layers)):
            x = self._layers[i](x)

        return x

class Classifier(models.Model):
    def __init__(self):

        super(Classifier, self).__init__(name="classifier")

        self._conv_block1 = ConvBlock(64)
        self._conv_block2 = ConvBlock(128, use_bn=True, use_dropout=True)
        self._conv_block3 = ConvBlock(256, use_bn=True, use_dropout=True)
        self._conv_block4 = ConvBlock(512)

        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(0.2)
        self._class = MLP()

    def call(self, inputs):
        x = self._conv_block1(inputs)
        x = self._conv_block2(x)
        x = self._conv_block3(x)
        x = self._conv_block4(x)
        x = self._flatten(x)
        x = self._dropout(x)

        classes = self._class(x)

        return classes


class WGCN_GP(models.Model):
    def __init__(self, num_batches, from_ckpt=False):

        super(WGCN_GP, self).__init__(name='wgcn_gp')

        self.d_loss = tf.keras.metrics.Mean(name="d_loss")
        #self.c_loss = tf.keras.metrics.Mean(name="c_loss")
        #self.c_accuracy = tf.keras.metrics.CategoricalAccuracy(name="c_acc")
        #self.g_losses = []
        self.g_loss = tf.keras.metrics.Mean(name="g_loss")
        self.t_loss = tf.keras.metrics.Mean(name="t_loss")

        self.discriminator = Discriminator()
        if from_ckpt:
            self.discriminator.build(input_shape=(None, 32, 32, 1))
            self.discriminator.load_weights(hps.savedir + 'discriminator' + ".h5")

        self.num_gens = hps.num_gens
        self.generators = []
        for i in range(self.num_gens):
            self.generators.append(Generator(i))
            #self.g_losses.append(tf.keras.metrics.Mean(name="g"+str(i)+"_loss"))
        if from_ckpt:
            for i in range(self.num_gens):
                self.generators[i].build(input_shape=(None, hps.noise_dim))
                self.generators[i].load_weights(hps.savedir + "gen{}".format(i) + ".h5")

        self.classifier = Classifier()
        self.classifier.build(input_shape=(None, 32, 32, 1))
        self.classifier.load_weights(hps.savedir + 'classifier' + ".h5")

        self.latent_dim = hps.noise_dim
        self.c_steps = hps.disc_iters_per_gen_iter
        self.gp_weight = hps.gp_weight
        self.batch_size = hps.batch_size
        self.num_classes = hps.num_classes
        self.c_loss_wt = hps.c_loss_weight
        #self.c_wt_decay = (self.c_loss_wt - hps.min_c_wt) / (hps.epochs * num_batches)
        #self.calculate_tvd_loss = True

    def compile(self, d_optimizer, g_optimizers, d_loss_fn, g_loss_fn, tvd_loss_fn):
        super(WGCN_GP, self).compile()
        self.d_opt = d_optimizer
        self.g_opts = g_optimizers
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.tvd_loss_fn = tvd_loss_fn

    def gradient_penalty(self, real_images, fake_images):
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    """def update_c_wt(self):
        self.c_loss_wt = self.c_loss_wt - self.c_wt_decay"""

    def train_step(self, real_data):
        real_images = real_data[0]
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

                fake_d = self.discriminator(fake_images, training=True)
                real_d = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_d, fake_img=fake_d)
                gp = self.gradient_penalty(real_images, fake_images)

                total_cost = d_cost + gp * self.gp_weight

            c_grads = tape.gradient(total_cost, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(
                zip(c_grads, self.discriminator.trainable_variables)
            )
            self.d_loss.update_state(total_cost)

        return_metrics["d_loss"] = self.d_loss.result()

        rvs = tf.random.normal(shape=(self.num_gens*self.batch_size, self.latent_dim))
        rvs = tf.split(rvs, self.num_gens, axis=0)

        with tf.GradientTape(persistent=True) as tape:
            fake_images = tf.nest.map_structure(
                lambda gen, rv: gen(rv, training=True),
                self.generators, rvs
            )
            fake_images = tf.concat(fake_images, axis=0)

            gen_d = self.discriminator(fake_images, training=True)
            gen_c = self.classifier(fake_images, training=True)
            gen_c = tf.nn.softmax(gen_c)
            gen_d = tf.split(gen_d, self.num_gens, axis=0)
            gen_c = tf.split(gen_c, self.num_gens, axis=0)

            gen_losses = tf.nest.map_structure(
                lambda discs: self.g_loss_fn(discs),
                gen_d
            )
            self.g_loss.update_state(tf.reduce_mean(gen_losses))
            return_metrics["g_loss"] = self.g_loss.result()

            tvd_loss = self.tvd_loss_fn(gen_c)
            self.t_loss.update_state(tvd_loss)
            return_metrics["t_loss"] = self.t_loss.result()
            return_metrics["net_objective"] = return_metrics["t_loss"] + return_metrics["d_loss"]

            tvd_loss = self.c_loss_wt * (1.0 - tvd_loss)
            gen_losses_c = tf.nest.map_structure(
                lambda g_loss: g_loss + tvd_loss,
                gen_losses
            )

        for i in range(self.num_gens):
            g_i_grad = tape.gradient(gen_losses_c[i], self.generators[i].trainable_variables)
            self.g_opts[i].apply_gradients(
                zip(g_i_grad, self.generators[i].trainable_variables)
            )

        #self.update_c_wt()

        return return_metrics

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss, self.t_loss]
