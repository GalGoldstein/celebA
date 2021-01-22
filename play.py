from __future__ import print_function
from run import Generator, Discriminator
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
import utils
from kmeans_pytorch import kmeans
import pandas as pd

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run2_30epochs')).to(device)
netD = torch.load(os.path.join('/home/student/HW3/celebA', 'netD_run2_30epochs')).to(device)

def get_attributes_file(path):
    attr_dict = dict()  # {image_id: -1/1 vector for 41 attributes}
    # txt_file = os.system(path)
    file = open(path, "r")
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        elif idx == 1:
            header = line.split()
        else:
            attr_dict[line[:6]] = line.split()[1:]

    return attr_dict, header

def plot_images(title, images, nrow=8):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images.cpu().detach()[:64], nrow=nrow,
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

def get_difference_vector_between_groups(A_name, B_name, A_idx, B_idx, latent_vectors):
    """
    :param A_name: representative name for group A
    :param B_name: representative name for group B
    :param A_idx: list of indices of group A members, in the latent vectors tensors
    :param B_idx: list of indices of group B members, in the latent vectors tensors
    :param latent_vectors: aka fixed_noise, to reproduce the same fake images
    :return: difference_vector: group_B - group_A
    """
    fake_images = netG(latent_vectors)
    A_images = fake_images[A_idx]
    B_images = fake_images[B_idx]

    # plot_images(f'both groups fake images', fake_images)
    # plot_images(f'{A_name} fake images', A_images)
    # plot_images(f'{B_name} fake images', B_images)

    A_average_vector = torch.mean(latent_vectors[A_idx], dim=0)
    B_average_vector = torch.mean(latent_vectors[B_idx], dim=0)
    difference_vector = B_average_vector - A_average_vector

    A_to_B_images = netG((latent_vectors[A_idx] + difference_vector).reshape([-1, latent_vectors.shape[1], 1, 1]))
    plot_images(f'{A_name} and {A_name}_to_{B_name} fake images', torch.cat((A_images, A_to_B_images),
                                                                            dim=0), len(A_images))

    B_to_A_images = netG((latent_vectors[B_idx] - difference_vector).reshape([-1, latent_vectors.shape[1], 1, 1]))
    plot_images(f'{B_name} and {B_name}_to_{A_name} fake images', torch.cat((B_images, B_to_A_images),
                                                                            dim=0), len(B_images))
    return difference_vector




if __name__ == "__main__":
    attr_path = '/datashare/list_attr_celeba.txt'
    attr_dict, header = get_attributes_file(attr_path)

    attr_df = pd.DataFrame.from_dict(attr_dict, orient='index', columns=header)
    for col in attr_df.columns:
        attr_df[col] = pd.to_numeric(attr_df[col])

    images_id_group_A = attr_df.index[attr_df['Male'] == 1].tolist()
    images_id_group_B = attr_df.index[attr_df['Male'] == -1].tolist()

    nz = 100
    images_amount = 64
    fixed_noise = torch.randn(images_amount, nz, 1, 1, device=device)  # used for plot the same fake images

    # for run1_5epochs
    # men = [0, 1, 11, 14, 26, 32, 41, 43, 52, 58]
    # women = [3, 9, 10, 12, 17, 19, 21, 24, 25, 28, 37, 38, 44, 47, 56, 57, 61, 62]

    # for run2_30epochs
    men = [8, 9, 21, 23, 24, 30, 33, 37, 38, 41]
    women = [1, 4, 5, 12, 14, 19, 25, 27, 28, 44, 59, 61]
    get_difference_vector_between_groups("men", "women", men, women, fixed_noise)

##################################################
# TODO Ideas: \
#  2. Vector aritmetic: try to differ between male and female, and get the difference vector
#  7. how to define the discrete z?
#  8. how to integrate the discrete and continuous z?



#  1. PCA to 2 dimensions, in order to see z latent spaces
#  3. train only on specific group (such as 'with glasses') and see if the results preserve it
#  4. kmeans, KNN,
#  5. Create the backword generator (image -> z, based on the generator with the current weights)
#  6. How can we use ATTRIBUTES, IDENTITY, LANDMARKS to get Z that present each characteristic?



# TODO Actions and tries: \
#  - run the model for longer time, to see if the fake images look better
#  - improve model hyper parameters?