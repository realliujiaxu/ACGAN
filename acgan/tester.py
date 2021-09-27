from stylegan2_pytorch.helper import noise_list, latent_to_w, styles_def_to_tensor, image_noise, add_attr
import torch
from stylegan2_pytorch import StyleGAN2


class Tester:

    def __init__(self,
                 gan: StyleGAN2,
                 image_size):
        self.GAN = gan
        self.image_size = image_size
        self.latent_dim = self.GAN.latent_dim

    def generate_images(self, attr_list):
        # 对只改变属性的一组图片，共享相同的noise和z（即这里的latents）
        noi = image_noise(1, self.image_size, device=0)
        latents = noise_list(1, 7, self.latent_dim, device=0)

        # 生成w
        w_space = latent_to_w(self.GAN.SE, latents)
        w_styles = styles_def_to_tensor(w_space)  # [1, 7, 512]
        w_styles = w_styles.expand(len(attr_list), -1, -1)

        # 拼接latents和attr_list
        moving_attrs = torch.stack(attr_list, dim=0)
        b, l, _ = w_styles.shape
        wattrs = moving_attrs.expand(b, l, -1)
        w_styles = torch.cat([w_styles, wattrs], dim=-1)

        return self.GAN.GE(w_styles, noi).clamp_(0., 1.)
