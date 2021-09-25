from torch.utils.data import DataLoader

import eval_factory
from acgan.model_factory import ClsRegFactory
from dataset_factory import DatasetType, get_dateset

if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

    # 测试CelebA数据集 回归器精度
    mode = "regressor"
    checkpiont = "/home/jiaxuliu/GANs/AttributeGAN-w/saves/celeba/regressor_5.pth"
    dataset = get_dateset(DatasetType.CELEBA_REG)

    # 测试CelebA数据集 分类器精度
    # mode = "classifier"
    # checkpiont = "/home/jiaxuliu/GANs/data/saves/celeba_cls/model_32.pth"
    # dataset = get_dateset(DatasetType.CELEBA_CLS)

    # 下面的流程是固定的，不用改
    model_factory = ClsRegFactory()
    model = model_factory.build(mode, checkpiont)
    eval_tool = eval_factory.get(mode)
    dataloader = DataLoader(dataset, 32, num_workers=4, shuffle=True)
    eval_tool.execute(model, dataloader)

