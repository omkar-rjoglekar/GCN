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
            #self.model.classifier.save_weights(self.save_path + "classifier" + ".h5")

            print("Saved models!")

class SuddenDeath(callbacks.Callback):
    def __init__(self):
        self.epoch_threshold = hps.ep_thres

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.epoch_threshold:
            print("stopping tvd loss")
            self.model.calculate_tvd_loss = False

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


class TVDistance(metrics.Metric):
    def __init__(self, name="tv_distance", **kwargs):
        super(TVDistance, self).__init__(name=name, **kwargs)

        self.batch_metrics = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.pointer = 0

    def update_state(self, y0, y1, sample_weight=None):
        tvd = tf.reduce_mean(0.5*tf.reduce_sum(tf.math.abs(y0 - y1), axis=-1))
        self.batch_metrics = self.batch_metrics.write(self.pointer, tvd)
        self.pointer += 1

    def result(self):
        metrics_tensor = self.batch_metrics.stack()
        return tf.reduce_mean(metrics_tensor)

    def reset_state(self):
        self.batch_metrics.close()
        self.batch_metrics = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.pointer = 0

