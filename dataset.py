import torch
import os
import json
import pickle
import platform
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import time
import torchvision.transforms.functional as TF
import utils


class CelebDataset(Dataset):
    """Visual Question Answering v2 dataset."""

    def __init__(self, images_path: str, create_imgs_tensors: bool = False,
                 read_from_tensor_files: bool = True, force_mem: bool = False):
        self.images = utils.load_images(images_path)  # all images as dictionary {'000001' : torch.tensor ...}

    def __getitem__(self, idx):
        image_id = str(idx + 1).zfill(6)
        return {'image_id': image_id, 'image_tensor': self.images[image_id]}

    def __len__(self):
        return len(self.images)



if __name__ == '__main__':

    # test dataset module
    # vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
    #                                questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
    #                                images_path='data/images',
    #                                phase='train', create_imgs_tensors=False, read_from_tensor_files=True)
    #
    # vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
    #                              questions_json_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
    #                              images_path='data/images',
    #                              phase='val', create_imgs_tensors=False, read_from_tensor_files=True)