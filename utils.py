from tensorflow.keras import callbacks, metrics
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

from hyperparameters import hps


class GCNMonitor(callbacks.Callback):
    def __init__(self):
        self.num_imgs = hps.save_img_num
        self.latent_dim = hps.noise_dim
        self.num_gens = hps.num_gens
        self.save_freq = hps.save_freq
        self.save_path = hps.gen_img_dir
        self.expt = hps.experiment_name

    def save_imgs(self, generated, gen_num, epoch):
        for i in range(self.num_imgs):
            img = generated[i]
            img = tf.keras.preprocessing.image.array_to_img(img)
            filename = "gen{gen_num}/epoch_{epoch}_{i}.png".format(gen_num=gen_num, i=i, epoch=epoch+1)
            filename = self.save_path + filename
            img.save(filename)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == (self.save_freq - 1):
            random_latent_vectors = tf.random.normal(shape=(self.num_imgs, self.latent_dim))
            for i in range(self.num_gens):
                generated_imgs = self.model.generators[i].predict(random_latent_vectors)
                generated_imgs = (generated_imgs * 127.5) + 127.5
                self.save_imgs(generated_imgs, i, epoch)


class GCNCheckpointer(callbacks.Callback):
    def __init__(self):
        self.num_gens = hps.num_gens
        self.save_freq = hps.save_freq
        self.save_path = hps.savedir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == (self.save_freq - 1):
            print("\nEpoch number {epoch} saving models to {path}".format(epoch=epoch+1, path=self.save_path))
            for i in range(self.num_gens):
                self.model.generators[i].save_weights(self.save_path+"gen"+str(i)+".h5")
            self.model.discriminator.save_weights(self.save_path + "discriminator" + ".h5")
            self.model.classifier.save_weights(self.save_path + "classifier" + ".h5")

            print("Saved models!")


def get_dataset(train=True):
    (train_images, train_labels), (test_images, test_labels) = load_data()
    if not train:
        train_images = test_images
        train_labels = test_labels
    train_images = (train_images - 127.5) / 127.5
    train_images = tf.image.resize(tf.expand_dims(train_images, -1), hps.img_shape[:2])
    train_labels = tf.one_hot(train_labels, hps.num_classes, 1, 0)

    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    if train:
        ds = ds.shuffle(100000, reshuffle_each_iteration=True).batch(hps.batch_size, drop_remainder=True)
    else:
        ds = ds.batch(hps.batch_size, drop_remainder=True)

    return ds


class JSDivergence(metrics.Metric):
    def __init__(self, name="js_divergence", **kwargs):
        super(JSDivergence, self).__init__(name=name, **kwargs)

        self.kld10 = metrics.KLDivergence()
        self.kld01 = metrics.KLDivergence()

    def update_state(self, y0, y1, sample_weight=None):
        self.kld01.update_state(y0, y1)
        self.kld10.update_state(y1, y0)

    def result(self):
        return 0.5 * (self.kld10.result() + self.kld01.result())

    def reset_state(self):
        self.kld01.reset_state()
        self.kld10.reset_state()
