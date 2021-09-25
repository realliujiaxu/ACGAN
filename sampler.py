import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from stylegan2_pytorch.scene import SceneDataset
from stylegan2_pytorch.regressor import build_regressor
import torch

class Generator(nn.Module):
    def __init__(self, n_attr):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 50, True),
            nn.ReLU(True),
            nn.Linear(50, 50, True),
            nn.Linear(50, n_attr),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_attr):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_attr, 50, True),
            nn.Linear(50, 1, True)
        )

    def forward(self, x):
        return self.model(x)

def wgangp(prediction, traget_is_real):
    if traget_is_real:
        return -prediction.mean()
    else:
        return prediction.mean()

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).cuda()

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def train(dataroot, n_epoch=50, batch_size=64):
    dataset = SceneDataset(dataroot, 128)
    dataloader = DataLoader(dataset, 64, num_workers=8)
    sampler = Generator(len(dataset.selected_idx)).cuda()
    discriminator = Discriminator(len(dataset.selected_idx)).cuda()

    # G_opt = optim.SGD(sampler.parameters(), lr=0.0001, momentum=0.9)
    # D_opt = optim.SGD(discriminator.parameters(), lr=0.0001, momentum=0.9)

    G_opt = optim.Adam(sampler.parameters(), lr=0.0001, betas=(0.5, 0.999))
    D_opt = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    criterion = GANLoss(gan_mode='wgangp')

    for epoch in range(1, n_epoch+1):
        for i, (_, attrs) in enumerate(tqdm(dataloader)):
            z = torch.rand(batch_size, 20).cuda()
            attrs = attrs.cuda()
            fake_attrs = sampler(z)

            # train discriminator
            set_requires_grad(discriminator, True)
            pred_attrs = discriminator(attrs)
            pred_fake_attrs = discriminator(fake_attrs.detach())
            # D_loss = wgangp(pred_attrs, True) + wgangp(pred_fake_attrs, False)
            D_loss = criterion(pred_attrs, True) + criterion(pred_fake_attrs, False)
            D_opt.zero_grad()
            D_loss.backward()
            D_opt.step()

            # train generator
            set_requires_grad(discriminator, False)
            pred_fake_attrs = discriminator(fake_attrs)
            # G_loss = wgangp(pred_fake_attrs, True)
            G_loss = criterion(pred_fake_attrs, True)
            G_opt.zero_grad()
            G_loss.backward()
            G_opt.step()

            if i % 100 == 0:
                print('G loss {:.4f}, D_loss {:.4f}'.format(G_loss.item(), D_loss.item()))
                print('fake attribute', fake_attrs.cpu().detach().numpy(), 'real attribute', attrs.cpu().detach().numpy())


if __name__ == "__main__":
    train('../../data/Scene/')
