from numpy import linspace
import os
import random
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch


# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return torch.cat(vectors, dim=0)


# create a plot of generated images
def plot_generated(interpolated_points, G):
    generated_images = G(interpolated_points)
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_images.cpu().detach(),
                                             padding=2, nrow=len(generated_images), normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def test_interpolated():
    """test the interpolated images module"""
    # Set random seed for reproducibility
    manualSeed = 999
    manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    nz = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run2_30epochs')).to(device)
    p1 = torch.randn(1, nz, 1, 1, device=device)
    p2 = torch.randn(1, nz, 1, 1, device=device)
    plot_generated(interpolate_points(p1, p2), netG)


if __name__ == '__main__':
    test_interpolated()
