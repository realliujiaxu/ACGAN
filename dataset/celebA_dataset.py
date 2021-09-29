from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
from functools import partial
from dataset.util import *


class CelebADataset(Dataset):
    """人脸属性二分类数据集"""

    def __init__(self, root, image_size, transparent=False, aug_prob=0.):
        """
        Args:
            root (string): Directory to scene Transient Attributes Database(http://transattr.cs.brown.edu/)
            transform (callable): transform to be applied
                on a sample.
        """

        super().__init__()
        attr_file = os.path.join(root, 'Anno', 'list_attr_celeba.txt')
        self.imgs = self.read_attr(attr_file)
        self.root = root
        self.image_size = image_size

        self.selected_idx = list(range(40))

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __getitem__(self, idx):
        img_name, attr_values = self.imgs[idx]
        attr_values = attr_values[self.selected_idx]
        path = os.path.join(self.root, 'img_align_celeba', img_name)

        img = Image.open(path)
        return self.transform(img), attr_values

    def read_attr(self, attr_file):
        import re
        imgs = []
        with open(attr_file) as fin:
            lines = fin.readlines()
            self.attr_names = lines[1].strip().split(' ')
            for l in lines[2:-1]:
                splits = re.split(" +", l.strip())
                img_name = splits[0]
                attrs = [int(i) for i in splits[1:]]
                imgs.append([img_name, torch.tensor(attrs, dtype=torch.float)])
        return imgs

    def __len__(self):
        return len(self.imgs)
