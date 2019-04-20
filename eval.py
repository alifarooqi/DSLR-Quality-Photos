import argparse
import os
from glob import glob
from scipy.misc import imread
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description="testing options")

parser.add_argument("phone_model", type=str, help="phone model to test")
parser.add_argument("--test_dir", type=str, default="D:/FYPdenoising/dslr/test")

config = parser.parse_args()

config.test_dir = os.path.join(config.test_dir, config.phone_model, "patches/*")

files = sorted(glob(config.test_dir))
num_samples = int(len(files) / 3)

gts = np.array([])
inputs = np.array([])
outputs = np.array([])
for i in range(1):
    print(files[i * 3], files[i * 3 + 1], files[i * 3 + 2])
    gts.append(imread(files[i * 3], mode="RGB"))
    inputs.append(imread(files[i * 3 + 1], mode="RGB"))
    outputs.append(imread(files[i * 3 + 2], mode="RGB"))

tf.convert_to_tensor(gts+0., dtype=tf.float32)
tf.convert_to_tensor(inputs+0., dtype=tf.float32)
tf.convert_to_tensor(outputs+0., dtype=tf.float32)

print(np.shape(tf.square(gaussian_blur(gts) - gaussian_blur(outputs))))
loss = tf.reduce_mean(tf.square(gaussian_blur(gts) - gaussian_blur(outputs)))
print(loss)
