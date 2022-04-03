import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm

import utils
import model
from hyperparameters import hps


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def evaluate(c_model, g_models):
    real_image_ds = utils.get_dataset(False)

    distance = utils.JSDivergence()
    fid = tf.keras.metrics.Mean()
    for (real_imgs, _) in real_image_ds:
        real_logits = c_model.predict(real_imgs)
        real_dist = tf.nn.softmax(real_logits)

        rvs = tf.random.normal(shape=(hps.num_gens * hps.batch_size//2, hps.noise_dim))
        rvs = tf.split(rvs, hps.num_gens, axis=0)
        fake_images = tf.nest.map_structure(
            lambda rv, gen_i: gen_i.predict(rv),
            rvs, g_models
        )
        fake_images = tf.concat(fake_images, axis=0)
        fake_logits = c_model.predict(fake_images)
        fake_dists = tf.nn.softmax(fake_logits)
        fake_dists_list = tf.split(fake_dists, hps.num_gens, axis=0)
        frechet_dist = calculate_fid(fake_dists.numpy(), real_dist.numpy())

        fid.update_state(frechet_dist)
        distance.update_state(fake_dists_list[1], fake_dists_list[0])

    return fid.result(), distance.result()


def save_im(g_models):

    rvs = tf.random.normal(shape=(hps.num_gens*hps.batch_size, hps.noise_dim))
    rvs = tf.split(rvs, hps.num_gens, axis=0)

    fake_images = tf.nest.map_structure(
        lambda rv, gen_i: gen_i.predict(rv),
        rvs, g_models
    )
    fake_images = tf.concat(fake_images, axis=0)
    generated_imgs = (fake_images * 127.5) + 127.5
    for i in range(hps.num_gens*hps.batch_size):
        img = generated_imgs[i]
        img = tf.keras.preprocessing.image.array_to_img(img)
        filename = "gen/{i}.png".format(i=i)
        filename = hps.gen_img_dir + filename
        img.save(filename)


if __name__ == "__main__":
    tf.keras.backend.clear_session()

    classifier = model.Classifier()
    classifier.build((None, 32, 32, 1))
    classifier.load_weights(hps.savedir + "classifier" + ".h5")

    generators = []
    for i in range(hps.num_gens):
        gen = model.Generator(i)
        gen.build((None, hps.noise_dim))
        gen.load_weights(hps.savedir + "gen{}".format(i) + ".h5")
        generators.append(gen)

    FID, dist = evaluate(classifier, generators)
    save_im(generators)

    print("Generator mean FID = {}".format(FID))
    print("Generators' image distance = {}".format(dist))
