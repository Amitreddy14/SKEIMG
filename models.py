import argparse
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate of optimizer")
parser.add_argument("--b", type=float, default=0.5, help="beta of optimizer")
parser.add_argument("--lam1", type=float, default=0.99, help="coefficient of perceptual loss")
parser.add_argument("--lam2", type=float, default=0.01, help="coefficient of contextual loss")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--gf_dim", type=int, default=64, help="dimension of gen filters in first conv layer.")
parser.add_argument("--df_dim", type=int, default=64, help="dimension of discrim filters in first conv layer.")
opt = parser.parse_args()
