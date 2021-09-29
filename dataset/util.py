import torch
import torchvision
import torch.nn as nn
from random import random


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