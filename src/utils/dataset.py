import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np

from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, image_path='../../data/DIS5K/DIS-TR/im', mask_path='../../data/DIS5K/DIS-TR/im',
                 image_transform=None, mask_transform=None, load_on_mem=False):
        self.images = glob(os.path.join(image_path, '*'))
        self.images.sort()

        self.masks = glob(os.path.join(mask_path, '*'))
        self.masks.sort()
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.load_on_mem = load_on_mem

        if self.load_on_mem:
            self.load_data_on_mem()

    def __len__(self):
        return len(self.masks)

    def load_data_on_mem(self):
        for idx in tqdm(range(self.__len__())):
            self.images[idx], self.masks[idx] = self.transform_dataset(self.images[idx], self.masks[idx])

    def transform_dataset(self, image, mask):
        image, mask = Image.open(image), Image.open(mask)
        mask = transforms.Grayscale()(mask)
        image, mask = np.array(image), np.array(mask)

        if self.image_transform:
            transformed = self.image_transform(image=image)
            image = transformed['image']
        if self.mask_transform:
            transformed = self.mask_transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        image = transforms.ToTensor()(image).float()
        mask = transforms.ToTensor()(mask).float()
        return image, mask

    def __getitem__(self, idx):
        if self.load_on_mem:
            image, mask = self.images[idx], self.masks[idx]
        else:
            image, mask = self.transform_dataset(self.images[idx], self.masks[idx])

        return {'image': image, 'mask': mask}