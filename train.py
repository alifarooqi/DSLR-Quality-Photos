import argparse
from utils import *
from dataloader import *
from model import *
import tensorflow as tf

parser = argparse.ArgumentParser(description="training options")

parser.add_argument("--num_epochs", type=int, help="number of epochs to train network for", default=100000)
parser.add_argument("--use_bn", help="use batch nor or not. default is True", type=bool, default=True)
parser.add_argument("--batch_size", type=int, help="batch size for each epoch", default=50)
parser.add_argument("--test_size", type=float, help="fraction for test size", default=0.10)
parser.add_argument("--dataset_dir", type=str, help="directory for the SSID dataset only",
                    default="dataset/SIDD_Small_sRGB_Only/Data/*/*.PNG")
parser.add_argument("--res", type=int, help="resolution of images", default=200)
parser.add_argument("--augment", help="use data augmentation or not. default is True", type=bool, default=True)
parser.add_argument("--w_content_loss", type=float, help="weightage for the content loss based on the VGG19",
                    default=0.5)
parser.add_argument("--w_adversarial_loss", type=float, help="weightage for discriminator guesses", default=1.0)
parser.add_argument("--w_pixel_loss", type=float, help="weightage for loss based on difference from GT", default=1.0)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.002)
parser.add_argument("--vgg_dir", type=str, help="directory for trained VGG 19 model",
                    default="vgg_pretrained/imagenet-vgg-verydeep-19.mat")
parser.add_argument("--content_layer", type=str, help="content layer to use in VGG 19 net", default="relu5_4")
parser.add_argument("--checkpoint_dir", type=str, help="directory for storing model checkpoints", default="./checkpoints")
parser.add_argument("--testing_dir", type=str, help="directory for storing testing images", default="./testing")
parser.add_argument("--num_files_to_load", type=int, help="number of images to load", default=None)
parser.add_argument("--load_checkpoint", type=bool, help="load checkpoint or not", default=False)
parser.add_argument("--test_mode", type=bool, help="testing mode or not", default=False)

config = parser.parse_args()

if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.Session()
    data_loader = DataLoader(config)
    model = Model(sess, config, data_loader)
    model.train(config.load_checkpoint)
