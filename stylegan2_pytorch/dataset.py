from torch.utils import data
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
from functools import partial
import torch.nn as nn
from random import random, randint
from pathlib import Path

EXTS = ['jpg', 'jpeg', 'png']

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

def exists(val):
    return val is not None

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'*/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

        print(f'loaded dataset of {len(self)} images')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pass


class CelebADataset(data.Dataset):
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

class CelebADatasetWithName(data.Dataset):
    """人脸属性二分类数据集，返回图片名"""

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
        return self.transform(img), attr_values, img_name

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


class SceneryDataset(data.Dataset):
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
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
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


class AttributeDataset(Dataset):
    """Dataset for ACGAN training"""

    def __init__(self, folder, image_size, selected_idx=None, transparent = False, aug_prob = 0., dataset='scene'):
        super().__init__(folder, image_size, transparent, aug_prob)
        
        if selected_idx is None:
            self.selected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        else:
            self.selected_idx = selected_idx
        if dataset == 'scene':
            attr_file = os.path.join(folder, 'annotations', 'annotations.csv')
            attrs = []
            with open(attr_file) as fin:
                for line in fin:
                    splits = line.strip().split('\t')
                    attr_values = [float(i.split(',')[0]) for i in splits[1:]]
                    attrs.append(attr_values)
            self.attrs = attrs
        elif dataset == 'celeba':
            attr_file = os.path.join(folder, 'Anno', 'normed-attr.txt')
            attrs = []
            with open(attr_file) as fin:
                lines = fin.readlines()
                for l in lines:
                    splits = l.strip().split(',')
                    attr_values = [float(i) for i in splits[1:]]
                    attrs.append(attr_values)
            self.attrs = attrs
        else:
            raise NotImplementedError(f'dataset should be scene or celeba, {dataset} if not implemented!')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        attr = torch.tensor(self.attrs[index % len(self.attrs)])
        return self.transform(img), attr[self.selected_idx]