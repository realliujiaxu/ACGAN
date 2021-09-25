import numpy as np
import torch
from stylegan2_pytorch import StyleGAN2

__all__ = ['Stylegan2Generator']


class Stylegan2Generator:
    """Defines the generator class of StyleGAN2.

    Different from conventional GAN, StyleGAN2 introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self):
        self.model_specific_vars = ['truncation.truncation']
        self.num_layers = 7
        self.latent_space_dim = 512
        self.model_path = "models/face256random.pt"
        self.resulotion = 256
        self.build()
        self.load()

    def build(self):
        self.model = StyleGAN2(lr=0, lr_mlp=0.1, ttur_mult=1.5, image_size=self.resulotion, network_capacity=16,
                               fmap_max=512, transparent=False, fq_layers=[], fq_dict_size=256, attn_layers=[],
                               fp16=False, cl_reg=False, no_const=False, rank=0)
        self.model = self.model.cuda()

    def load(self):
        print(f'Loading pytorch model from `{self.model_path}`.')
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict['GAN'])
        print(f'Successfully loaded!')

    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        latent_space_type = latent_space_type.upper()

        latent_codes = torch.randn(num, self.latent_space_dim).cuda()

        if latent_space_type == 'W':
            latent_codes = self.model.SE(latent_codes)
        elif latent_space_type == 'WP':
            latent_codes = self.model.SE(latent_codes)
            latent_codes = self.ws_to_wps(latent_codes)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes, torch.FloatTensor(1, self.resulotion, self.resulotion, 1).uniform_(0., 1.)

    def synthesize(self,
                   latent_codes,
                   noi,
                   latent_space_type='Z',
                   generate_style=False,
                   generate_image=True):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        results = {}
        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape
        # Generate from Z space.
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = latent_codes
            ws = self.model.SE(zs)
            wps = self.ws_to_wps(ws)
            results['z'] = latent_codes
            results['w'] = self.get_value(ws)
            results['wp'] = self.get_value(wps)
        # Generate from W space.
        elif latent_space_type == 'W':
            ws = latent_codes
            wps = self.ws_to_wps(ws)

        return self.model.GE(wps, noi).clamp_(0., 1.)

    def ws_to_wps(self, ws):
        return ws[:, None, :].expand(-1, self.num_layers, -1)

    def postprocess(self, images):
        images = np.clip(images * 255 + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        return images
