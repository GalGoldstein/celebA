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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

netG = torch.load(os.path.join('/home/student/HW3/celebA', 'netG_run1_5epochs')).to(device)
netD = torch.load(os.path.join('/home/student/HW3/celebA', 'netD_run1_5epochs')).to(device)

nz = 100
amount = 64
fixed_noise = torch.randn(amount, nz, 1, 1, device=device)
fake = netG(fixed_noise)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake.cpu().detach()[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# cluster fake images by their z random vectors
num_clusters = 5
fixed_noise = fixed_noise.reshape([amount, nz])  # reshape the noise
cluster_ids_x, cluster_centers = kmeans(
    X=fixed_noise, num_clusters=num_clusters, distance='euclidean', device=device)

# plot images in the same cluster
# for cluster in range(num_clusters):
#     fake_cluster_images = fake[(cluster_ids_x == cluster).nonzero().flatten()]
#     plt.figure(figsize=(8, 8))
#     plt.axis("off")
#     plt.title(f'cluster={cluster} Images')
#     plt.imshow(np.transpose(vutils.make_grid(fake_cluster_images.cpu().detach()[:64],
#                                              padding=2, normalize=True).cpu(), (1, 2, 0)))
#     plt.show()

men = [0, 1, 11, 14, 26, 32, 41, 43, 52, 58]
women = [3, 9, 10, 12, 17, 19, 21, 24, 25, 28, 37, 38, 44, 47, 56, 57, 61, 62]
men_images = fake[men]
women_images = fake[women]
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("men fake images")
plt.imshow(np.transpose(vutils.make_grid(men_images.cpu().detach()[:64],padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()


plt.axis("off")
plt.title("women fake images")
plt.imshow(np.transpose(vutils.make_grid(women_images.cpu().detach()[:64],padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()


men_z_average = torch.mean(fixed_noise[men], dim=0)
women_z_average = torch.mean(fixed_noise[women], dim=0)
difference = women_z_average - men_z_average

men_to_women = netG((fixed_noise[men]+difference).reshape([-1,nz,1,1]))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("men & men_to_women")
total_images_1 = torch.cat((men_images,men_to_women),dim=0)
plt.imshow(np.transpose(vutils.make_grid(total_images_1.cpu().detach(),
                                            padding=2, nrow=len(men_images) , normalize=True).cpu(), (1, 2, 0)))
plt.show()


women_to_men = netG((fixed_noise[women]-difference).reshape([-1,nz,1,1]))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("women & women_to_men")
total_images_2 = torch.cat((women_images,women_to_men),dim=0)
plt.imshow(np.transpose(vutils.make_grid(total_images_2.cpu().detach(),
                                         padding=2, nrow=len(women_images) , normalize=True).cpu(), (1, 2, 0)))
plt.show()

# TODO Ideas: \
#  1. PCA to 2 dimensions, in order to see z latent spaces
#  2. Vector aritmetic: try to differ between male and female, and get the difference vector
#  3. train only on specific group (such as 'with glasses') and see if the results preserve it
#  4. kmeans, KNN,
#  5. Create the backword generator (image -> z, based on the generator with the current weights)
#  6. How can we use ATTRIBUTES, IDENTITY, LANDMARKS in order to improve the model?
#  7. how to define the discrete z?
#  8. how to integrate the discrete and continous z?


# TODO Actions and tries: \
#  - run the model for longer time, to see if the fake images look better
#  - improve model hyper parameters?