from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
from functools import partial
from dataset.util import *


class SceneryDataset(Dataset):
    """自然场景数据集"""

    def __init__(self, root, image_size, transparent=False, aug_prob=0.):
        """
        Args:
            root (string): Directory to scene Transient Attributes Database(http://transattr.cs.brown.edu/)
            transform (callable): transform to be applied
                on a sample.
        """

        super().__init__()
        csv_file = os.path.join(root, 'annotations', 'annotations.csv')
        self.paths = self.read_tsv(csv_file)
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
        img_name, attr_values = self.paths[idx]
        attr_values = attr_values[self.selected_idx]
        path = os.path.join(self.root, 'unaligned', img_name)

        img = Image.open(path)
        return self.transform(img), attr_values, img_name

    def read_tsv(self, csv_file):
        imgs = []
        with open(csv_file) as fin:
            for line in fin:
                splits = line.strip().split('\t')
                img_name = splits[0]
                attr_values = [float(i.split(',')[0]) for i in splits[1:]]
                imgs.append((img_name, torch.tensor(attr_values)))
        return imgs

    def __len__(self):
        return len(self.paths)
