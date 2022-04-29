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
    def __init__(self, from_ckpt=False):

        super(WGCN_GP, self).__init__(name='wgcn_gp')

        self.d_loss = tf.keras.metrics.Mean(name="d_loss")
        self.c_loss = tf.keras.metrics.Mean(name="c_loss")
        self.c_accuracy = tf.keras.metrics.CategoricalAccuracy(name="c_acc")
        self.g_losses = []

        self.discriminator = Discriminator()
        if from_ckpt:
            self.discriminator.build(shape=(None)+hps.img_shape)
            self.discriminator.load_weights(hps.savedir + 'discriminator' + ".h5")

        self.num_gens = hps.num_gens
        self.generators = []
        for i in range(self.num_gens):
            self.generators.append(Generator(i))
            self.g_losses.append(tf.keras.metrics.Mean(name="g"+str(i)+"_loss"))
        if from_ckpt:
            for i in range(self.num_gens):
                self.generators[i].build(shape=(None, hps.noise_dim))
                self.generators[i].load_weights(hps.savedir + "gen{}".format(i) + ".h5")

        self.classifier = Classifier()
        if from_ckpt:
            self.classifier.build(shape=(None)+hps.img_shape)
            self.classifier.load_weights(hps.savedir + 'classifier' + ".h5")

        self.latent_dim = hps.noise_dim
        self.c_steps = hps.disc_iters_per_gen_iter
        self.gp_weight = hps.gp_weight
        self.batch_size = hps.batch_size
        self.num_classes = hps.num_classes

    def compile(self, d_optimizer, g_optimizers, c_optimizer, d_loss_fn, g_loss_fn):
        super(WGCN_GP, self).compile()
        self.d_opt = d_optimizer
        self.g_opts = g_optimizers
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.classifier.compile(optimizer=c_optimizer,
                                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

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

    def train_step(self, real_data):
        real_images = real_data[0]
        real_labels = real_data[1]
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

        with tf.GradientTape() as tape:
            preds = self.classifier(real_images, training=True)
            loss = self.classifier.compiled_loss(real_labels, preds, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.classifier.trainable_variables)
        self.classifier.optimizer.apply_gradients(
            zip(grads, self.classifier.trainable_variables)
        )
        self.c_loss.update_state(loss)
        self.c_accuracy.update_state(real_labels, preds)
        return_metrics["c_loss"] = self.c_loss.result()
        return_metrics["c_acc"] = self.c_accuracy.result()

        rvs = tf.random.normal(shape=(self.num_gens*self.batch_size, self.latent_dim))
        rvs = tf.split(rvs, self.num_gens, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            fake_images = tf.nest.map_structure(
                lambda gen, rv: gen(rv, training=True),
                self.generators, rvs
            )
            fake_images = tf.concat(fake_images, axis=0)

            gen_d = self.discriminator(fake_images, training=True)
            gen_d = tf.split(gen_d, self.num_gens, axis=0)

            gen_losses = tf.nest.map_structure(
                lambda discs: self.g_loss_fn(discs),
                gen_d
            )

        for i in range(self.num_gens):
            g_i_grad = tape.gradient(gen_losses[i], self.generators[i].trainable_variables)
            self.g_opts[i].apply_gradients(
                zip(g_i_grad, self.generators[i].trainable_variables)
            )
            self.g_losses[i].update_state(gen_losses[i])
            return_metrics["g" + str(i) + "_loss"] = self.g_losses[i].result()

        return return_metrics

    @property
    def metrics(self):
        return [self.d_loss, self.c_loss, self.c_accuracy] + self.g_losses
