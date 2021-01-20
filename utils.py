import torch
import os
import platform
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


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


def resize(size, path):
    """

    :param size: resize images to 3*size*size
    :param path: path to images folder
    """
    if not os.path.exists(path + '_pt'):
        os.makedirs(path + '_pt')

    files_names = os.listdir(path)
    resize = transforms.Resize(size=(size, size))
    for file_name in files_names:
        img_path = os.path.join(path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = resize(image)
        image = TF.to_tensor(image)
        save_path = os.path.join(path + '_pt', file_name.split('.')[0] + '.pt')
        torch.save(image, save_path)  # if we wish float16 >>> torch.save(image.to(dtype=torch.float16), save_path))


if __name__ == '__main__':
    running_on_linux = 'Linux' in platform.platform()
    path = 'img_sample' if not running_on_linux else 'img_align_celeba'
    resize(size=64, path=path)
