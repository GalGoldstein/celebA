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
from dataset import CelebDataset
import platform
from torch.utils.data import DataLoader
import utils


def weights_init(m):  # TODO understand and change names
    """ Inspired from the article """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, nz_discrete):
        super(Generator, self).__init__()
        self.nz = nz
        self.nz_discrete = nz_discrete
        self.de_cnn = nn.Sequential(  # input: z  -> output: fake image 3x128x128
            nn.ConvTranspose2d(self.nz + self.nz_discrete, G_nfeatures * 16,  kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(G_nfeatures * 16),
            nn.ReLU(True),
            # size. (G_nfeatures*16) x 4 x 4
            nn.ConvTranspose2d(G_nfeatures * 16, G_nfeatures * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures * 8),
            nn.ReLU(True),
            # size. (G_nfeatures*8) x 8 x 8
            nn.ConvTranspose2d(G_nfeatures * 8, G_nfeatures * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures * 4),
            nn.ReLU(True),
            # size. (G_nfeatures*4) x 16 x 16
            nn.ConvTranspose2d(G_nfeatures * 4, G_nfeatures * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures * 2),
            nn.ReLU(True),
            # size. (G_nfeatures*2) x 32 x 32
            nn.ConvTranspose2d(G_nfeatures * 2, G_nfeatures, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures),
            nn.ReLU(True),
            # size. (G_nfeatures) x 64 x 64
            nn.ConvTranspose2d(G_nfeatures, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.de_cnn(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(  # input: image 3x128x128 -> output: binary decision whether the image is fake
            nn.Conv2d(nc, D_nfeatures, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size. ndfx64x64
            nn.Conv2d(D_nfeatures, D_nfeatures * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # size. (D_nfeatures*2)x32x32
            nn.Conv2d(D_nfeatures * 2, D_nfeatures * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # size. D_nfeatures*4x16x16
            nn.Conv2d(D_nfeatures * 4, D_nfeatures * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # size. D_nfeatures*8x8x8
            nn.Conv2d(D_nfeatures * 8, D_nfeatures * 16, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(D_nfeatures * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # size. D_nfeatures*16x3x3
            nn.Conv2d(D_nfeatures * 16, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.cnn(input)


if __name__ == "__main__":
    manualSeed = 999  # TODO change to other seed
    print("Fixed Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    running_on_linux = 'Linux' in platform.platform()
    dataroot = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA',
                                                                         'img_align_celeba_pt')
    resize_to = 128
    run = "run5"
    workers = 2  # Number of workers for dataloader
    batch_size = 128  # Batch size during training
    nc = 3  # Number of channels in the training images. For color images this is 3
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    nz_discrete = 40  # Size of z latent vector (i.e. size of generator input)
    G_nfeatures = resize_to  # Size of feature maps in generator
    D_nfeatures = resize_to  # Size of feature maps in discriminator
    epochs = 6  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #  RUN ONLY ONCE: preprocessing and convert images to tensors
    # preprocessing_path = 'img_sample' if not running_on_linux \
    #     else '/home/student/HW3/celebA/img_align_celeba'
    # utils.images_preprocessing(size=128, path=preprocessing_path)

    celeb_dataset = CelebDataset(dataroot)
    dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    netG = Generator(nz, nz_discrete).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_continuous_noise = torch.randn(resize_to, nz, 1, 1, device=device)
    fixed_discrete_noise = torch.randint(0, 2, (resize_to, nz_discrete, 1, 1), device=device)
    fixed_noise = torch.cat((fixed_continuous_noise, fixed_discrete_noise), dim=1)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Start Training...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data['images_tensor'].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            loss_D_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            loss_D_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            continuous_noise = torch.randn(b_size, nz, 1, 1, device=device)
            discrete_noise = torch.randint(0, 2, (b_size, nz_discrete, 1, 1), device=device)
            noise = torch.cat((continuous_noise, discrete_noise), dim=1)

            fake = netG(noise)  # Generate fake image batch with G
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)  # Classify all fake batch with D
            loss_D_fake = criterion(output, label)  # Calculate D's loss on the all-fake batch
            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake  # Add the gradients from the all-real and all-fake batches
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            loss_G = criterion(output, label)  # Calculate G's loss based on this output
            loss_G.backward()
            D_G_z = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\t,Loss_D: %.4f\tLoss_G: %.4f\t,D(x): %.4f\tD(G(z)): %.4f'
                      % (epoch + 1, epochs, i, len(dataloader),
                         loss_D.item(), loss_G.item(), D_x, D_G_z))

            # Save Losses for plotting later
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    torch.save(netG, os.path.join('/home/student/HW3/celebA', f'netG_{run}_add_discrete_{epochs}epochs'))
    torch.save(netD, os.path.join('/home/student/HW3/celebA', f'netD_{run}_add_discrete_{epochs}epochs'))
    # torch.cuda.empty_cache()

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Visualization of G’s progression
    # %%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    html_file = HTML(ani.to_jshtml()).data
    text_file = open(os.path.join('/home/student/HW3/celebA', "html_output_file.html"), "w")
    text_file.write(f'{html_file},{run}')
    text_file.close()

    # Real Images vs. Fake Images
    real_batch = next(iter(dataloader))

    # Plot real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['images_tensor'].to(device)[:64],
                                             padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
