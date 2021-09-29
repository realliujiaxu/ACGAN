import os
from PIL import Image
from dataset.util import *
from dataset.dataset import Dataset


class AttributeDataset(Dataset):
    """ACGAN训练用的数据集，返回每张图片对应的属性值"""

    def __init__(self, folder, image_size, selected_idx=None, transparent=False, aug_prob=0., dataset='scene'):
        super().__init__(folder, image_size, transparent, aug_prob)

        if selected_idx is None:
            self.selected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
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
