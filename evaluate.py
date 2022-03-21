import tensorflow as tf

import utils
import model
from hyperparameters import hps

"""
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
"""


def evaluate():
    classifier = model.Classiminator()
    classifier.build((None, 32, 32, 1))
    classifier.load_weights(hps.savedir + "classiminator" + ".h5")
    real_image_ds = utils.get_dataset(False)
    generators = []
    for i in range(hps.num_gens):
        gen = model.Generator(i)
        gen.build((None, hps.noise_dim))
        gen.load_weights(hps.savedir + "gen{}".format(i)+".h5")
        generators.append(gen)

    distance = utils.JSDivergence()
    jsd_real0 = utils.JSDivergence()
    jsd_real1 = utils.JSDivergence()
    jsd_real = utils.JSDivergence()
    for (real_batch, _) in real_image_ds:
        _, real_logits = classifier.predict(real_batch)
        real_dist = tf.nn.softmax(real_logits)

        rvs = tf.random.normal(shape=(hps.num_gens*hps.batch_size, hps.noise_dim))
        rvs = tf.split(rvs, hps.num_gens, axis=0)
        fake_images = tf.nest.map_structure(
            lambda rv, gen_i: gen_i.predict(rv),
            rvs, generators
        )
        fake_images = tf.concat(fake_images, axis=0)
        _, fake_logits = classifier.predict(fake_images)
        fake_dist = tf.nn.softmax(fake_logits)
        fake_dist = tf.split(fake_dist, hps.num_gens, axis=0)

        distance.update_state(fake_dist[1], fake_dist[0])
        jsd_real0.update_state(fake_dist[0], real_dist)
        jsd_real1.update_state(fake_dist[1], real_dist)

        rvs = tf.random.normal(shape=(hps.batch_size, hps.noise_dim))
        rvs = tf.split(rvs, hps.num_gens, axis=0)
        fake_images = tf.nest.map_structure(
            lambda rv, gen_i: gen_i.predict(rv),
            rvs, generators
        )
        fake_images = tf.concat(fake_images, axis=0)
        _, fake_logits = classifier.predict(fake_images)
        fake_dist = tf.nn.softmax(fake_logits)

        jsd_real.update_state(fake_dist, real_dist)

    return distance.result(), jsd_real0.result(), jsd_real1.result(), jsd_real.result()


if __name__ == "__main__":
    cross, real0, real1, real = evaluate()
    print("Mean FID values:\n")
    print("Cross JSD = {}".format(cross))
    print("Real and Gen JSD = {}".format(real))
    print("Real and Gen 0 JSD = {}".format(real0))
    print("Real and Gen 1 JSD = {}".format(real1))
