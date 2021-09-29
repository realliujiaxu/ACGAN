from enum import Enum

from dataset.dataset import AttributeDataset, CelebADataset, SceneryDataset

celeba_path = '/home/jiaxuliu/GANs/data/CombinedFace/celeba'
secene_path = "/home/jiaxuliu/GANs/data/CombinedFace/SceneDataset"


class DatasetType(Enum):
    CELEBA_CLS = 0
    CELEBA_REG = 1
    SCENE = 2


def get_dateset(type):
    if type == DatasetType.CELEBA_CLS: # celeba数据集+类别标注，-1或1
        return CelebADataset(celeba_path, 256)
    elif type == DatasetType.CELEBA_REG: # celeba数据集+量化属性标注，范围在[0, 1]
        return AttributeDataset(celeba_path, 256, dataset='celeba')
    elif type == DatasetType.SCENE: # scene数据集
        return SceneryDataset(secene_path, 256)
