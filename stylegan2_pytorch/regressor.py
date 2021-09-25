import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def build_regressor(n_cls=40, pretrain=False):
    model_conv = models.resnet50(pretrained=pretrain)
    # for param in model_conv.parameters():
    #     param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs // 2),
        nn.ReLU(),
        nn.Linear(num_ftrs // 2, n_cls)
    )

    # Observe that all parameters are being optimized
    return model_conv