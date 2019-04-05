import tensorflow as tf
from scipy.misc import imresize
import numpy as np
from vgg19 import *

mean_RGB = np.array([123.68, 116.779, 103.939])


def resize_img(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = imresize(x, shape, interp='bicubic')
    return y


def load_files(files, res, test_mode):
    if not test_mode:
        loaded = [resize_img(scipy.misc.imread(filename, mode="RGB"), (res, res)) for filename
                  in files]
    else:
        loaded = [resize_img(scipy.misc.imread(filename, mode="RGB"), (1500, 1500)) for filename
                  in files]
    return loaded


def convlayer(input, output, ksize, stride, name, use_bn, activation="leaky_relu"):
    temp = tf.layers.conv2d(input, output, ksize, stride, padding="same", name=name,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    if use_bn:
        temp = tf.layers.batch_normalization(temp, name="BN_" + name)
    if activation:
        if activation == "leaky_relu":
            temp = tf.nn.leaky_relu(temp, name="relu_" + name)
        elif activation == "tanh":
            temp = tf.nn.tanh(temp, name="tanh_" + name)
        else:
            temp = activation(temp)
    return temp


def resblock(input, output, resblock_num, use_bn):
    rb_conv1 = convlayer(input, output, 3, 1, ("rb_%d_conv_1" % resblock_num), use_bn)
    rb_conv2 = convlayer(rb_conv1, output, 3, 1, ("rb_%d_conv_2" % resblock_num), use_bn)
    return rb_conv2 + input


def preprocess(img):
    return img / 255


def postprocess(img):
    return np.round(np.clip(img * 255, 0, 255)).astype(np.uint8)


def calc_pixel_loss(gt, generated):
    return tf.nn.l2_loss(gt - generated)


def get_content_loss(vgg_dir, gt, generated, content_layer):
    enhanced_vgg = net(vgg_dir, gt * 255)
    gt_vgg = net(vgg_dir, generated * 255)
    return tf.reduce_mean(tf.square(enhanced_vgg[content_layer] - gt_vgg[content_layer]))
