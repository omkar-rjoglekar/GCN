from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

import utils
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


def evaluate():
    classifier = load_model(hps.savedir + "classiminator")
    real_image_ds = utils.get_dataset(False)
    generators = []
    for i in range(hps.num_gens):
        generators.append(load_model(hps.savedir + "gen{}".format(i)))

    distance_01 = tf.keras.metrics.KLDivergence()
    distance_10 = tf.keras.metrics.KLDivergence()
    fid_real0 = []
    fid_real1 = []
    fid_real = []
    for (real_batch, _) in real_image_ds:
        #print(real_batch)
        _, real_logits = classifier.predict(real_batch)

        rvs = tf.random.normal(shape=(hps.num_gens * hps.batch_size, hps.noise_dim))
        rvs = tf.split(rvs, hps.num_gens, axis=0)
        fake_images = tf.nest.map_structure(
            lambda rv, gen: gen.predict(rv),
            rvs, generators
        )
        fake_images = tf.concat(fake_images, axis=0)
        _, fake_logits = classifier.predict(fake_images)
        real_fid = calculate_fid(real_logits, fake_logits)
        fake_logits = tf.split(fake_logits, hps.num_gens, axis=0)

        real_fid0 = calculate_fid(real_logits, fake_logits[0].numpy())
        real_fid1 = calculate_fid(real_logits, fake_logits[1].numpy())

        fake_dist0 = tf.nn.softmax(fake_logits[0])
        fake_dist1 = tf.nn.softmax(fake_logits[1])

        fid_real0.append(real_fid0)
        fid_real1.append(real_fid1)
        fid_real.append(real_fid)

        distance_10.update_state(fake_dist1, fake_dist0)
        distance_01.update_state(fake_dist0, fake_dist1)

    return 0.5 * (distance_01.result() + distance_10.result()), np.mean(fid_real0), np.mean(fid_real1), np.mean(fid_real)

if __name__ == "__main__":
    cross, real0, real1, real = evaluate()
    print("Mean FID values:\n")
    print("Cross JSD = {}".format(cross))
    print("Real and Gen FID = {}".format(real))
    print("Real and Gen 0 FID = {}".format(real0))
    print("Real and Gen 1 FID = {}".format(real1))
