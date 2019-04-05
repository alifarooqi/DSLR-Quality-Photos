from utils import *
from vgg19 import *
import time
import os


class Model(object):
    def __init__(self, session, config, dataloader):
        self.sess = session
        self.config = config
        self.data_loader = dataloader
        self.noisy_train = dataloader.noisy_train
        self.noisy_test = dataloader.noisy_test
        self.gt_train = dataloader.gt_train
        self.gt_test = dataloader.gt_test

        self.generator_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                           name="generator_input")
        self.gt_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                    name="gt_input")
        self.enhanced_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                          name="enhanced_input")
        self.generator_in_test = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name="test_input")

        self.generated = self.generator(self.generator_in)
        self.discriminator_gt = self.discriminator(self.gt_in)
        self.discriminator_enhanced = self.discriminator(self.generated)

        self.generator_test = self.generator(self.generator_in_test)

        print("setting up loss functions")
        self.d_loss = -tf.reduce_mean(tf.log(self.discriminator_gt) + tf.log(1. - self.discriminator_enhanced))
        self.g_loss = self.config.w_adversarial_loss * -tf.reduce_mean(
            tf.log(self.discriminator_enhanced)) + self.config.w_pixel_loss * calc_pixel_loss(self.gt_in,
                                                                                              self.generated) + self.config.w_content_loss * get_content_loss(
            self.config.vgg_dir, self.gt_in, self.generated, self.config.content_layer)

        t_vars = tf.trainable_variables()
        discriminator_vars = [var for var in t_vars if 'discriminator' in var.name]
        generator_vars = [var for var in t_vars if 'generator' in var.name]

        self.discriminator_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss,
                                                                                               var_list=discriminator_vars)
        self.generator_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.g_loss,
                                                                                           var_list=generator_vars)

        tf.global_variables_initializer().run(session=self.sess)
        self.saver = tf.train.Saver(tf.trainable_variables())

    def generator(self, feature_in):
        print("Setting up the generator network")
        use_bn = self.config.use_bn
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            conv1 = convlayer(feature_in, 32, 3, 1, "conv_1", use_bn)
            conv2 = convlayer(conv1, 64, 3, 1, "conv_2", use_bn)
            conv3 = convlayer(conv2, 128, 3, 1, "conv_3", use_bn)
            conv4 = convlayer(conv3, 128, 3, 1, "conv_4", use_bn)
            rb1 = resblock(conv4, 128, 1, use_bn)
            rb2 = resblock(rb1, 128, 2, use_bn)
            rb3 = resblock(rb2, 128, 3, use_bn)
            conv5 = convlayer(rb3, 128, 3, 1, "conv_5", use_bn)
            conv6 = convlayer(conv5, 64, 3, 1, "conv_6", use_bn)
            conv7 = convlayer(conv6, 3, 3, 1, "conv_7", False, "tanh")
            output = tf.clip_by_value(tf.add(conv7, feature_in), 0.0, 1.0)
            return output

    def discriminator(self, feature_in):
        print("setting up the discriminator network")
        use_bn = self.config.use_bn
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            conv1 = convlayer(feature_in, 48, 3, 2, "conv_1", use_bn)
            conv2 = convlayer(conv1, 96, 3, 2, "conv_2", use_bn)
            conv3 = convlayer(conv2, 192, 3, 2, "conv_3", use_bn)
            conv4 = convlayer(conv3, 96, 3, 1, "conv_4", use_bn)
            flat = tf.contrib.layers.flatten(conv4)
            fc1 = tf.layers.dense(flat, units=96)
            logits = tf.layers.dense(fc1, units=1)
            prob = tf.nn.sigmoid(logits)
            return prob

    def train(self, load=False):

        if load:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Overall training starts from beginning")

        start = time.time()
        for epoch in range(0, self.config.num_epochs):
            if epoch == 1:
                print("1 epoch completed")
            noisy_batch, gt_batch = self.data_loader.get_batch()
            _, enhanced_batch = self.sess.run([self.generator_solver, self.generated],
                                              feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})
            _ = self.sess.run(self.discriminator_solver,
                              feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})

            if epoch % 200 == 0:
                g_loss = self.sess.run(self.g_loss,
                                       feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})
                print("Iteration %d, runtime: %.3f s, generator loss: %.6f" % (
                    epoch, time.time() - start, g_loss))
                self.save()

    def test(self):
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return [], []
        noisy_batch, _ = self.data_loader.get_batch()
        enhanced_batch = self.sess.run(self.generator_test, feed_dict={self.generator_in_test: noisy_batch})
        return noisy_batch, enhanced_batch

    def save(self):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, "my_model"), write_meta_graph=False)

    def load(self):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("Loading checkpoints from ", checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, "my_model"))
            return True
        else:
            return False
