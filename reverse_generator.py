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
    """
        Get latent vectors for given images (Like going with the generator in the ooposite direction)
        Done by MSE minimization min ||G(z)-Images||
    :param G: Generator, Pre-trained
    :param images: batch of images of size [batch_size, 3, 64, 64] or similar size
    :param nz: z dimension used to train G
    :param niter: number of iterations for MSE optimization
    :return: z that if we apply G(z) it will be close to 'images'
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    z_approx = torch.randn(images.size(0), nz, 1, 1, device=device)
    z_approx.requires_grad_(True)
    optimizer = optim.Adam([z_approx])

    for i in range(niter):
        optimizer.zero_grad()
        g_z_approx = G(z_approx)
        loss = criterion(g_z_approx, images)
        if i and i % 100 == 0:
            print(f"[Iter {i}]\t mse_g_z: {float(loss)}")

        # back propagation
        loss.backward()
        optimizer.step()

    return z_approx


def test_reverse_generator():
    """test the reverse generator module"""
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    running_on_linux = 'Linux' in platform.platform()
    dataroot = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA',
                                                                         'img_align_celeba_pt')
    nz = 100
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    celeb_dataset = CelebDataset(dataroot)
    dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run2_30epochs')).to(device)
    # get real images
    images = next(iter(dataloader))['images_tensor'].to(device)
    # find the z that approximate the real images
    z_approx = reverse_generator(G=netG, images=images, niter=20000)
    # try to generate the real images from their z
    generated_images = netG(z_approx)

    # Plot the real images
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(images.to(device), padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    # Plot the generated images from z
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("Generated Images from approximated z")
    plt.imshow(
        np.transpose(vutils.make_grid(generated_images.to(device)[:64], padding=5, normalize=True).detach().cpu(),
                     (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test_reverse_generator()