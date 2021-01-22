from __future__ import print_function
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
import platform
from torch.utils.data import DataLoader
import utils
from run import Generator, Discriminator
from dataset import CelebDataset


def reverse_generator(G, images, nz=100, niter=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    z_approx = torch.randn(images.size(0), nz, 1, 1, device=device)
    optimizer = optim.Adam([z_approx])

    for i in range(niter):
        optimizer.zero_grad()

        g_z_approx = G(z_approx)
        loss = criterion(g_z_approx, images)
        if i % 100 == 0:
            print(f"[Iter {i}]\t mse_g_z: {float(loss)}")

        # back propagation
        loss.backward()
        optimizer.step()

    return z_approx


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
running_on_linux = 'Linux' in platform.platform()
dataroot = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA', 'img_align_celeba_pt')
nz = 100
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
celeb_dataset = CelebDataset(dataroot)
dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run1_5epochs')).to(device)
images = next(iter(dataloader))['images_tensor']

reverse_generator(G=netG, images=images)

breakpoint()
