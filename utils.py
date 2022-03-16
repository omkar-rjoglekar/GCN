from tensorflow.keras import callbacks
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf
import os

from hyperparameters import hps

class GCNMonitor(callbacks.Callback):
    def __init__(self):
        self.num_imgs = hps.save_img_num
        self.latent_dim = hps.noise_dim
        self.num_gens = hps.num_gens
        self.save_freq = hps.save_freq
        self.save_path = hps.gen_img_dir

    def save_imgs(self, generated, gen_num, epoch):
        for i in range(self.num_imgs):
            img = generated[i]#.numpy()
            img = tf.keras.preprocessing.image.array_to_img(img)
            filename = "generated_img_gen{gen_num}_{i}_{epoch}.png".format(gen_num=gen_num, i=i, epoch=epoch)
            filename = os.path.join(self.save_path, filename)
            img.save(filename)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == (self.save_freq - 1):
            random_latent_vectors = tf.random.normal(shape=(self.num_imgs, self.latent_dim))
            for i in range(self.num_gens):
                generated_imgs = self.model.generators[i].predict(random_latent_vectors)
                generated_imgs = (generated_imgs * 127.5) + 127.5
                self.save_imgs(generated_imgs, i, epoch)

def get_dataset():
    (train_images, train_labels), (_, _) = load_data()
    train_images = (train_images - 127.5) / 127.5
    train_images = tf.image.resize(tf.expand_dims(train_images, -1), hps.img_shape[:2])
    train_labels = tf.one_hot(train_labels, hps.num_classes, 1, 0)

    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.shuffle(100000, reshuffle_each_iteration=True).batch(hps.batch_size, drop_remainder=True)
    #print(ds)

    return ds

