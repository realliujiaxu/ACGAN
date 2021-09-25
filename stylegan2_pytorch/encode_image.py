import os.path

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image
from stylegan2_pytorch.perceptual_model import VGG16_for_Perceptual
import torch.optim as optim
from PIL import Image

from stylegan2_pytorch.stylegan2_pytorch import styles_def_to_tensor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def image_reader(img_path, resize=None):
    with open(img_path, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
    if resize != None:
        image = image.resize((resize, resize))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image


def caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d):
    # calculate MSE Loss
    mse_loss = MSE_Loss(synth_img, img)  # (lamda_mse/N)*||G(w)-I||^2

    # calculate Perceptual Loss
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img)  # (1,3,256,256)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = 0
    perceptual_loss += MSE_Loss(synth_0, real_0)
    perceptual_loss += MSE_Loss(synth_1, real_1)
    perceptual_loss += MSE_Loss(synth_2, real_2)
    perceptual_loss += MSE_Loss(synth_3, real_3)

    return mse_loss, perceptual_loss


def encode_image(img_path, regressor, generator, device=device, resolution=512, iteration=800, result_dir='encoding/'):
    image_size = 256
    latent_dim = 512
    num_attr = 40
    num_layers = 7

    generator.eval()
    img = image_reader(img_path, resize=image_size)
    img = img.to(device)

    MSE_Loss = nn.MSELoss(reduction="mean")
    img_p = img.clone()  # Perceptual loss 用画像
    upsample2d = torch.nn.Upsample(scale_factor=256/resolution, mode='bilinear')
    img_p = upsample2d(img_p)

    perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(device)
    w_styles = torch.zeros((1, 7, latent_dim), requires_grad=True, device=device)
    # attrs = torch.ones((1, num_attr), device=device) * 0.5
    # attrs.requires_grad_(True)
    attrs = regressor(img)
    wattrs = attrs[:, None].expand(1, 7, -1)
    # noi = torch.FloatTensor(1, image_size, image_size, 1).uniform_(0., 1.).to(device=device)
    noi = torch.zeros((1, image_size, image_size, 1), requires_grad=False, device=device) + 0.5
    noi.requires_grad_(True)

    latents = [[torch.zeros((1, latent_dim-num_attr), requires_grad=True, device=device), 1] for _ in range(7)]
    optimizer = optim.Adam({w_styles}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    noi_optimizer = optim.Adam({noi}, lr=0.0002, betas=(0.9, 0.999), eps=1e-8)

    print("Start")
    loss_list = []
    for i in range(1, iteration+1):
        optimizer.zero_grad()

        synth_img = generator(torch.cat([w_styles, wattrs], dim=-1), noi)
        # synth_img = (synth_img + 1.0) / 2.0
        mse_loss, perceptual_loss = caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d)
        loss = mse_loss + perceptual_loss
        # loss += + torch.mean(torch.square(noi))*50
        loss.backward()

        optimizer.step()
        noi_optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p = perceptual_loss.detach().cpu().numpy()
        loss_m = mse_loss.detach().cpu().numpy()

        loss_list.append(loss_np)
        if i % 100 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(i, loss_np, loss_m, loss_p))
            # save_image(synth_img.clamp(0, 1), result_dir+"/{}.png".format(i))
            # np.save("loss_list.npy",loss_list)
            # np.save("latent_W/{}.npy".format(name), dlatent.detach().cpu().numpy())
    return w_styles, attrs, noi, loss_np