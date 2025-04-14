import argparse
import numpy as np

from matplotlib import pyplot as plt

from models import *
from evaluation import *
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--z_dim", type=int, default=100, help="dimension of z sampler.")
opt = parser.parse_args()

# Alternate training
def alt_train(model, X_train):

    d_loss_list = []
    g_loss_list = []

    # Adversarial ground truths
    valid = np.ones((opt.batch_size, 1))
    fake = np.zeros((opt.batch_size, 1))

    X_train = X_train.astype('float32') / 255

    for epoch in range(opt.epochs):