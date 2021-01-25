from numpy import linspace
import os
import random
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch
import run
from run import Generator, Discriminator
from PIL import Image
from torchvision import transforms
from reverse_generator import reverse_generator

""" Tow points interpolation """


# uniform interpolation between two points in latent space
def interpolate_two_points(p1, p2, n_steps=10):
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
    manualSeed = run.manualSeed
    print("Fixed Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = torch.load(run.netG_path).to(device)
    for i in range(20):
        p1_continuous = torch.randn(1, run.z_ncontinuous, 1, 1, device=device)
        p1_discrete = torch.randint(0, 2, (1, run.z_ndiscrete, 1, 1), device=device)
        p1 = torch.cat((p1_continuous, p1_discrete), dim=1)

        p2_continuous = torch.randn(1, run.z_ncontinuous, 1, 1, device=device)
        p2_discrete = torch.randint(0, 2, (1, run.z_ndiscrete, 1, 1), device=device)
        p2 = torch.cat((p2_continuous, p2_discrete), dim=1)
        plot_generated(interpolate_two_points(p1, p2), netG)


""" Three points interpolation """


def interpolate_three_points(p1, p2, p3, n_steps=10):
    # interpolate ratios between the points
    ratios1 = linspace(0, 1, num=n_steps)
    ratios2 = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio1 in ratios1:
        for ratio2 in ratios2:
            if ratio1 + ratio2 > 1:
                continue
            v = (1.0 - ratio1 - ratio2) * p1 + ratio1 * p2 + ratio2 * p3
            vectors.append(v)
    return torch.cat(vectors, dim=0)


def plot_generated_three_points(interpolated_points, G):
    generated_images = G(interpolated_points)
    plt.figure(figsize=(15, 15))

    indexes = [j + 1 + 10 * i for i in range(10) for j in range(10) if j >= i]
    for i in range(len(generated_images)):
        plt.subplot(10, 10, indexes[0])
        del indexes[0]
        plt.axis('off')
        plt.imshow(np.transpose(vutils.make_grid(generated_images[i].cpu(), normalize=True).detach().numpy(),
                                (1, 2, 0)))
    plt.show()


def test_interpolated_three_points():
    """test the interpolated images module for three points"""
    # Set random seed for reproducibility
    manualSeed = run.manualSeed
    print("Fixed Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = torch.load(run.netG_path).to(device)
    for i in range(10):
        p1_continuous = torch.randn(1, run.z_ncontinuous, 1, 1, device=device)
        p1_discrete = torch.randint(0, 2, (1, run.z_ndiscrete, 1, 1), device=device)
        p1 = torch.cat((p1_continuous, p1_discrete), dim=1)

        p2_continuous = torch.randn(1, run.z_ncontinuous, 1, 1, device=device)
        p2_discrete = torch.randint(0, 2, (1, run.z_ndiscrete, 1, 1), device=device)
        p2 = torch.cat((p2_continuous, p2_discrete), dim=1)

        p3_continuous = torch.randn(1, run.z_ncontinuous, 1, 1, device=device)
        p3_discrete = torch.randint(0, 2, (1, run.z_ndiscrete, 1, 1), device=device)
        p3 = torch.cat((p3_continuous, p3_discrete), dim=1)
        plot_generated_three_points(interpolate_three_points(p1, p2, p3), netG)


""" Interpolate Tamir, Guy, Itay """


def interpolate_staff():
    files_names = ['tamir.jpg', 'itay.jpg', 'guy.jpg']
    size = 64
    transform = transforms.Compose([transforms.Resize(size=(size, size)),
                                    transforms.ToTensor(),  # move to tensor and normalize to [0,1]
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalize to [-1,1]
    images = list()
    for file_name in files_names:
        img_path = file_name
        image = Image.open(img_path).convert('RGB')
        images.append(transform(image))
    images_tensor = torch.stack(images, dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run2_30epochs')).to(device)
    z = reverse_generator(G=netG, images=images_tensor.to(device), nz=100, niter=20000)
    nz = 100
    shape = (1, nz, 1, 1)
    p1, p2, p3 = z[0].reshape(shape), z[1].reshape(shape), z[2].reshape(shape)
    plot_generated_three_points(interpolate_three_points(p1, p2, p3), netG)


if __name__ == '__main__':
    test_interpolated()
    # test_interpolated_three_points()
    # interpolate_staff()
    pass
