from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from dataset import CelebDataset
import platform
from torch.utils.data import DataLoader
from
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    running_on_linux = 'Linux' in platform.platform()
    dataroot = 'img_sample_pt' if not running_on_linux else 'img_align_celeba_pt' # Root directory for dataset
    workers = 2  # Number of workers for dataloader
    batch_size = 128  # Batch size during training
    nc = 3  # Number of channels in the training images. For color images this is 3
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    num_epochs = 5  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.

    #  run only once: preprocessing and convert images to tensors
    preprocessing_path = 'img_sample' if not running_on_linux else 'img_align_celeba'
    utils.images_preprocessing(size=64, path=preprocessing_path)

    # load_images(path + '_pt')
    celeb_dataset = CelebDataset(dataroot)
    dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    batch = next(iter(dataloader))
