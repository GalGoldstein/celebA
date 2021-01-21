import torch
import os
import platform
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def download_data():
    """
        downloads the data and unzips, delete zip when finish
    """
    print("Downloading celeba.zip ...")
    os.system('wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip')
    print("Unzipping celeba.zip ...")
    os.system('unzip celeba.zip')
    print("Deleting file celeba.zip ...")
    os.system('rm celeba.zip')


def images_preprocessing(size, path):
    """
    :param size: resize images to 3*size*size
    :param path: path to images folder
    """
    if not os.path.exists(path + '_pt'):
        os.makedirs(path + '_pt')

    files_names = os.listdir(path)
    transform = transforms.Compose([transforms.Resize(size=(size, size)),
                                    transforms.ToTensor(),  # move to tensor and normalize to [0,1]
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalize to [-1,1]

    for file_name in files_names:
        img_path = os.path.join(path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        save_path = os.path.join(path + '_pt', file_name.split('.')[0] + '.pt')
        torch.save(image, save_path)  # if we wish float16 >>> torch.save(image.to(dtype=torch.float16), save_path))


def load_images(path):
    """
        load all images from path to dictionary
    :param path: path to .pt images
    :return: all images as dictionary {'000001' : torch.tensor ...}
    """
    files_names = sorted(os.listdir(path))#[:1000] # TODO delete [:1000] and sorted
    all_images = dict()
    for file_name in files_names:
        image = torch.load(os.path.join(path, file_name))
        all_images[file_name.split('.')[0]] = image
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
    return all_images


if __name__ == '__main__':
    # original images size == (178, 218)
    running_on_linux = 'Linux' in platform.platform()
    path = 'img_sample' if not running_on_linux else 'img_align_celeba'
    images_preprocessing(size=64, path=path)
    # load_images(path + '_pt')
