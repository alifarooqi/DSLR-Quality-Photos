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

for i in range(1):
    gt_img = np.float32((imread(files[i * 3], mode="RGB")))
    input_img = np.float32((imread(files[i * 3 + 1], mode="RGB")))
    output_img = np.float32((imread(files[i * 3 + 2], mode="RGB")))
    loss = np.mean(np.square(gaussian_blur(gt_img) - gaussian_blur(output_img)))

