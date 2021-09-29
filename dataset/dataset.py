from torch.utils import data
from torchvision import transforms
from functools import partial
from pathlib import Path
from dataset.util import *

EXTS = ['jpg', 'jpeg', 'png']


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent=False, aug_prob=0.):
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
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

        print(f'loaded dataset of {len(self)} images')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pass
