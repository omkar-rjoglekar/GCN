import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import argparse

import utils
import model
from hyperparameters import hps


class Trainer:
    def __init__(self, from_checkpoint=False):
        self.d_opt = Adam(hps.disc_lr, hps.beta_1, hps.beta_2)
        self.c_opt = Adam(hps.class_lr)
        self.g_opts = []
        for i in range(hps.num_gens):
            self.g_opts.append(Adam(hps.gen_lr, hps.beta_1, hps.beta_2))

        self.real_dataset = utils.get_dataset()
        self.gcn_monitor_cbk = utils.GCNMonitor()
        self.checkpointer_cbk = utils.GCNCheckpointer()
        self.tensorboard_cbk = tf.keras.callbacks.TensorBoard(hps.logdir)

        self.num_epochs = hps.epochs

        self.gcn = model.WGCN_GP(from_checkpoint)

    def d_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def g_loss(self, disc):
        d_loss = -tf.reduce_mean(disc)
        return d_loss

    def train(self):
        self.gcn.compile(self.d_opt, self.g_opts, self.c_opt, self.d_loss, self.g_loss)

        results = self.gcn.fit(self.real_dataset,
                               epochs=self.num_epochs,
                               callbacks=[self.gcn_monitor_cbk,
                                          self.tensorboard_cbk,
                                          self.checkpointer_cbk],
                               verbose=1)

        self.gcn.discriminator.save_weights(hps.savedir + "discriminator" + ".h5")
        self.gcn.classifier.save_weights(hps.savedir + "classifier" + ".h5")
        for i in range(hps.num_gens):
            self.gcn.generators[i].save_weights(hps.savedir+"gen{}".format(i)+".h5")

        return results


if __name__ == "__main__":
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", "-c", action="store_true",
                        help="to use saved checkpoint to continue training")
    args = parser.parse_args()

    trainer = Trainer(args.from_checkpoint)
    history = trainer.train()
