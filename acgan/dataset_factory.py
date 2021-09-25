from enum import Enum

from stylegan2_pytorch.dataset import AttributeDataset, CelebADataset, SceneryDataset

celeba_path = '/home/jiaxuliu/GANs/data/CombinedFace/celeba'
secene_path = "/home/jiaxuliu/GANs/data/CombinedFace/SceneDataset"

class DatasetType(Enum):
    CELEBA_CLS = 0
    CELEBA_REG = 1
    SCENE = 2

def get_dateset(type):
    if type == DatasetType.CELEBA_CLS:
        return CelebADataset(celeba_path, 256)
    elif type == DatasetType.CELEBA_REG:
        return AttributeDataset(celeba_path, 256, dataset='celeba')
    elif type == DatasetType.SCENE:
        return SceneryDataset(secene_path, 256)