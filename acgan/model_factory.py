import torch
from torch import nn
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ClsRegFactory:

    def build_classifier(self, n_cls=40):
        model_conv = models.resnet50(pretrained=True)
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

    def build_regressor(self, n_cls=40, pretrain=False):
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

    def build(self, mode, checkpoint=None):
        if mode == "regressor":
            model = self.build_regressor()
            if checkpoint is not None:
                model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))["regressor"])
        else:
            model = self.build_classifier()
            if checkpoint is not None:
                model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))["model"])

        model = model.cuda()
        return model

class TrainedClsRegFactory:

    def __init__(self):
        self.clsreg_factory = ClsRegFactory()

    def get_trained_classifier(self):
        return self.clsreg_factory.build("classifier",  "/home/jiaxuliu/GANs/data/saves/celeba_cls/model_5.pth")

    def get_face_regressor(self):
        return self.clsreg_factory.build("regressor", "/home/jiaxuliu/GANs/AttributeGAN-w/saves/celeba/regressor_5.pth")

    def get_scene_regressor(self):
        return self.clsreg_factory.build("regressor", "/home/jiaxuliu/GANs/data/saves/scene/regressor_300.pth")

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    model_factory = ClsRegFactory()
    print(model_factory.build("Classifier"))
    print(model_factory.build("regressor", "/home/jiaxuliu/GANs/AttributeGAN-w/saves/celeba/regressor_1.pth"))