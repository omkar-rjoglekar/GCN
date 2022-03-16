import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import utils
import model
from hyperparameters import hps

class Trainer:
    def __init__(self):
        self.c_opt = Adam(hps.disc_lr, hps.beta_1, hps.beta_2)
        self.g_opts = []
        for i in range(hps.num_gens):
            self.g_opts.append(Adam(hps.gen_lr, hps.beta_1, hps.beta_2))

        self.real_dataset = utils.get_dataset()
        self.gcn_monitor_cbk = utils.GCNMonitor()
        #self.checkpointer_cbk = tf.keras.callbacks.ModelCheckpoint(hps.savedir, monitor='c_loss',
                                                                   #update_freq=hps.save_freq)
        self.tensorboard_cbk = tf.keras.callbacks.TensorBoard(hps.logdir)

        self.num_epochs = hps.epochs
        self.classifier_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.gcn = model.WGCN_GP()
        #self.gcn.build()

    def d_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def g_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def c_loss(self, real_img, fake_img, real_true, fake_true):
        preds = tf.concat((real_img, fake_img), axis=0)
        trues = tf.concat((real_true, fake_true), axis=0)

        return self.classifier_loss(trues, preds)

    def train(self):
        self.gcn.compile(self.c_opt, self.g_opts,
                         self.d_loss, self.c_loss,
                         self.g_loss)

        history = self.gcn.fit(self.real_dataset,
                               epochs=self.num_epochs,
                               callbacks=[self.gcn_monitor_cbk,
                                          self.tensorboard_cbk],
                               verbose=1)

        self.gcn.classiminator.save(hps.savedir+"classiminator")
        for i in range(hps.num_gens):
            self.gcn.generators[i].save(hps.savedir+"gen{}".format(i))

        return history

if __name__ == "__main__":
    trainer = Trainer()
    history = trainer.train()