import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import argparse

import utils
import model
from hyperparameters import hps


class Trainer:
    def __init__(self, from_checkpoint=False):
        self.c_opt = Adam(hps.disc_lr, hps.beta_1, hps.beta_2)
        self.g_opts = []
        for i in range(hps.num_gens):
            self.g_opts.append(Adam(hps.gen_lr, hps.beta_1, hps.beta_2))

        self.real_dataset = utils.get_dataset()
        self.gcn_monitor_cbk = utils.GCNMonitor()
        self.checkpointer_cbk = utils.GCNCheckpointer()
        self.tensorboard_cbk = tf.keras.callbacks.TensorBoard(hps.experiment_name+hps.logdir)

        self.num_epochs = hps.epochs
        self.classifier_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.gen_diff_loss = tf.keras.losses.KLDivergence()
        self.c_loss_weight = hps.c_loss_weight

        self.gcn = model.WGCN_GP(from_checkpoint)

    def d_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def g_loss(self, disc, classes):
        #c1, c2 = tf.split(classes, 2, axis=0)
        #c1 = tf.nn.softmax(c1)
        #c2 = tf.nn.softmax(c2)
        #c_loss1 = self.gen_diff_loss(c1, c2)
        #c_loss2 = self.gen_diff_loss(c2, c1)
        #c_loss = 0.5 * (c_loss1 + c_loss2)

        d_loss = -tf.reduce_mean(disc)

        return d_loss #- self.c_loss_weight * c_loss

    def c_loss(self, real_img, fake_img, real_true, fake_true):
        preds = tf.concat((real_img, fake_img), axis=0)
        trues = tf.concat((real_true, fake_true), axis=0)

        return self.classifier_loss(trues, preds)

    def train(self):
        self.gcn.compile(self.c_opt, self.g_opts,
                         self.d_loss, self.c_loss,
                         self.g_loss)

        results = self.gcn.fit(self.real_dataset,
                               epochs=self.num_epochs,
                               callbacks=[self.gcn_monitor_cbk,
                                          self.tensorboard_cbk,
                                          self.checkpointer_cbk],
                               verbose=1)

        self.gcn.classiminator.save_weights(hps.experiment_name+hps.savedir+"classiminator"+".h5")
        for i in range(hps.num_gens):
            self.gcn.generators[i].save_weights(hps.experiment_name+hps.savedir+"gen{}".format(i)+".h5")

        return results


if __name__ == "__main__":
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", "-c", action="store_true",
                        help="to use saved checkpoint to continue training")
    args = parser.parse_args()

    trainer = Trainer(args.from_checkpoint)
    history = trainer.train()
