import glob

import torch

from stylegan2_pytorch.exception import NanException, raise_if_nan
from stylegan2_pytorch.model import StyleGAN2
from stylegan2_pytorch.regressor import build_regressor

import os
import math
import json

from tqdm import tqdm
from math import floor, log2
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np

from torch.utils import data
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms
from stylegan2_pytorch.version import __version__
from PIL import Image
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import datetime
from stylegan2_pytorch.dataset import AttributeDataset
import time

from stylegan2_pytorch.helper import *

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from acgan.model_factory import TrainedClsRegFactory

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png']
CALC_FID_NUM_IMAGES = 12800

cls_factory = TrainedClsRegFactory()
classfier = cls_factory.get_trained_classifier()
sigmoid = nn.Sigmoid()

class Trainer():
    def __init__(
            self,
            selected_idx,
            reg_weight=40,
            latent_dim=512,
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            image_size=128,
            network_capacity=16,
            fmap_max=512,
            transparent=False,
            batch_size=4,
            mixed_prob=0.9,
            gradient_accumulate_every=1,
            lr=2e-4,
            lr_mlp=1.,
            ttur_mult=2,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            num_image_tiles=8,
            trunc_psi=0.6,
            fp16=False,
            cl_reg=False,
            fq_layers=[],
            fq_dict_size=256,
            attn_layers=[],
            no_const=False,
            aug_prob=0.,
            aug_types=['translation', 'cutout'],
            top_k_training=False,
            generator_top_k_gamma=0.99,
            generator_top_k_frac=0.5,
            dataset_aug_prob=0.,
            calculate_fid_every=None,
            is_ddp=False,
            rank=0,
            world_size=1,
            log=True,
            *args,
            **kwargs
    ):
        self.reg_weight = reg_weight
        self.latent_dim = latent_dim
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.starttime = datetime.datetime.now().__format__('%Y-%m-%d %H:%M:%S')
        self.logdir = f'runs/{self.name}'
        self.logger = SummaryWriter(f'{self.logdir}/{self.starttime}') if self.is_main else None

        self.regressor = build_regressor().cuda(self.rank)
        if 'face' in self.name:
            print('loading regressor from saves/celeba/regressor_5.pth')
            self.regressor.load_state_dict(torch.load('saves/celeba/regressor_5.pth')["regressor"])
        else:
            print('loading regressor from saves/scene/regressor_300.pth')
            self.regressor.load_state_dict(torch.load('saves/scene/regressor_300.pth')["regressor"])
        set_requires_grad(self.regressor, False)
        self.criterionL2 = nn.MSELoss()
        self.selected_idx = selected_idx
        self.n_attr = len(selected_idx)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(n_attr=self.n_attr, latent_dim=self.latent_dim, lr=self.lr, lr_mlp=self.lr_mlp,
                             ttur_mult=self.ttur_mult, image_size=self.image_size,
                             network_capacity=self.network_capacity, fmap_max=self.fmap_max,
                             transparent=self.transparent, fq_layers=self.fq_layers, fq_dict_size=self.fq_dict_size,
                             attn_layers=self.attn_layers, fp16=self.fp16, cl_reg=self.cl_reg, no_const=self.no_const,
                             rank=self.rank, *args, **kwargs)
        self.GAN.G.latent_dim -= self.n_attr

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.reg_weight = config.pop('reg_weight', 1)
        # self.latent_dim = config.pop('latent_dim', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'reg_weight': self.reg_weight, 'latent_dim': self.latent_dim, 'image_size': self.image_size,
                'network_capacity': self.network_capacity, 'fmap_max': self.fmap_max, 'transparent': self.transparent,
                'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers,
                'no_const': self.no_const}

    def set_data_src(self, folder):
        if 'face' in self.name:
            self.dataset = AttributeDataset(folder, self.image_size, transparent=self.transparent,
                                            aug_prob=self.dataset_aug_prob, dataset='celeba')
        else:
            self.dataset = AttributeDataset(folder, self.image_size, transparent=self.transparent,
                                            aug_prob=self.dataset_aug_prob)
        num_workers = default(self.num_workers, num_cores)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size,
                                     shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers=num_workers,
                                     batch_size=math.ceil(self.batch_size / self.world_size), sampler=sampler,
                                     shuffle=not self.is_ddp, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)
        self.n_attr = len(self.selected_idx)
        self.reg_weight = self.n_attr

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        # apply_path_penalty = self.steps > 5000 and self.steps % 32 == 0
        apply_path_penalty = False
        apply_cl_reg_to_generated = self.steps > 20000
        apply_attr_reg = self.steps % 1 == 0

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
                    noise = image_noise(batch_size, image_size, device=self.rank)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch, attrs = next(self.loader)
                image_batch = image_batch.cuda(self.rank)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id=0)

            self.GAN.D_opt.step()

        # train discriminator
        start_time = time.time()
        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        attrs_list = []

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()
        t_data = 0
        t_comp = 0
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            image_batch, attrs = next(self.loader)
            # attrs = torch.rand(batch_size, self.n_attr)
            image_batch = image_batch.cuda(self.rank)
            attrs = attrs.cuda(self.rank)
            attrs_list.append(attrs)

            iter_start_time = time.time()  # timer for data loading per iteration
            t_data += iter_start_time - iter_data_time

            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)
            b, l, _ = w_styles.shape
            attrs = attrs[:, None].expand(b, l, -1)
            w_styles = torch.cat([w_styles, attrs], dim=-1)

            generated_images = G(w_styles, noise)
            fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach=True, **aug_kwargs)

            image_batch.requires_grad_()
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = (F.relu(1 + real_output_loss) + F.relu(1 - fake_output_loss)).mean()
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

            t_comp += time.time() - iter_start_time
            iter_data_time = time.time()

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.GAN.D_opt.step()

        # train generator
        start_time = time.time()
        self.GAN.G_opt.zero_grad()
        attr_idx = 0
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            iter_start_time = time.time()  # timer for data loading per iteration
            attrs = attrs_list[attr_idx]
            attr_idx += 1

            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)
            b, l, _ = w_styles.shape
            wattrs = attrs[:, None].expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)

            generated_images = G(w_styles, noise)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = fake_output_loss.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            if apply_attr_reg:
                attr_reg = self.regressor(generated_images)
                gen_loss_reg = self.criterionL2(attr_reg[:, self.selected_idx], attrs) * self.reg_weight
                gen_loss += gen_loss_reg
                self.gen_loss_reg = gen_loss_reg

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id=2)
            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every
            t_comp += time.time() - iter_start_time
            if apply_path_penalty:
                print(f'apply_path_penalty compute time {t_comp}')

        self.GAN.G_opt.step()

        # logging Generator loss
        start_time = time.time()
        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')
        if self.is_main and apply_attr_reg and self.steps % 32 == 0:
            self.track(gen_loss_reg.item(), 'Reg')

        # calculate moving averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()  # update self.GAN.SE and self.GAN.GE

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            # This will not be excuted after 25000 iterations
            # which means GE is always EMA model after 25000 iterations.
            # Experiments show that EMA model give better results
            self.GAN.reset_parameter_averaging()

        # save from NaN errors
        start_time = time.time()
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if apply_path_penalty:
                print(f'apply_path_penalty compute time {t_comp}, data time {t_data}')
            if self.steps % 50 == 0:
                print(f'compute time {t_comp}, data time {t_data}')

            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0:
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')
        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num=0, trunc=1.0):
        ext = self.image_extension
        num_rows = self.num_image_tiles
        # save_dir = str(self.results_dir / self.name / str(num))
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        num_generate = num_rows ** 2
        latents = noise_list(num_generate, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_generate, image_size, device=self.rank)
        attrs = torch.rand(num_generate, self.n_attr).to(self.rank)
        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, attrs, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'),
                                     nrow=num_rows)

        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, attrs,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'),
                                     nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, attrs,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'),
                                     nrow=num_rows)

        # save attrs
        attrs_np = attrs.cpu().numpy()
        with open(str(self.results_dir / self.name / f'{str(num)}.txt'), 'w') as fout:
            for line_idx in range(num_rows):
                fout.write(f'{line_idx + 1}th line:\n')
                chunked_attr = attrs_np[line_idx * num_rows: line_idx * num_rows + num_rows]
                for i in range(len(attrs_np[0])):
                    c = [a[i] for a in chunked_attr]
                    fout.write('\t'.join(['{:.2f}'.format(k) for k in c]))
                    fout.write('\n')

        # continuing attribute
        n_attr = len(self.selected_idx)
        num_cols, num_rows = 11, n_attr
        noi = image_noise(1, self.image_size, device=self.rank)
        noi = noi.expand(num_cols * n_attr, -1, -1, -1)
        latents[0][0] = latents[0][0][:1].expand(num_cols * n_attr, -1)

        base_attr = torch.ones(n_attr).cuda(self.rank) * 0.5
        attr_list = []
        step = 1 / (num_cols - 1)
        for i in range(n_attr):
            for j in range(num_cols):
                new_attr = base_attr.detach().clone()
                new_attr[i] = j * step
                attr_list.append(new_attr)
        moving_attrs = torch.stack(attr_list, dim=0)

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noi, moving_attrs,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-moving.{ext}'),
                                     nrow=num_cols)
        with open(str(self.results_dir / self.name / f'{str(num)}-moving.txt'), 'w') as fout:
            fout.write(moving_attrs.__repr__())

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch, _ = next(self.loader)
            for k in range(real_batch.size(0)):
                torchvision.utils.save_image(real_batch[k, :, :, :],
                                             real_path + '{}.png'.format(k + batch_num * self.batch_size))

        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            n = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :],
                                             str(Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi=0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim - len(self.selected_idx), device=self.rank)
            attrs = torch.rand([2000, len(self.selected_idx)], device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z, attrs).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi=0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, attrs, trunc_psi=0.75, num_image_tiles=8):
        # w = map(lambda t: (S(t[0], attrs), t[1]), style)
        # w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        # w_styles = styles_def_to_tensor(w_truncated)
        latent_dim = self.GAN.G.latent_dim
        if not hasattr(self, 'avz'):
            z = noise(2000, latent_dim, device=self.rank)
            self.avz = torch.mean(z, dim=0)[None]
        style = [(trunc_psi * (z - self.avz) + self.avz, num_layers) for z, num_layers in style]
        w_space = latent_to_w(S, style)
        w_styles = styles_def_to_tensor(w_space)
        b, l, _ = w_styles.shape
        attrs = attrs[:, None].expand(b, l, -1)
        w_styles = torch.cat([w_styles, attrs], dim=-1)

        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    def cat_wstyle_attr(self, w_styles, attrs):
        b, l, _ = w_styles.shape
        attrs = attrs[:, None].expand(b, l, -1)
        w_styles = torch.cat([w_styles, attrs], dim=-1)
        return w_styles

    @torch.no_grad()
    def generate(self, S, G, style, noi, attrs, trunc_psi=0.75, num_image_tiles=8):
        w_space = latent_to_w(S, style)
        w_styles = styles_def_to_tensor(w_space)
        w_styles = self.cat_wstyle_attr(w_styles, attrs)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    def add_attr(self, attr_list, edit_attr_list, base_attr=None, degree=0.3):
        new_attr = attr_list[-1].clone() if base_attr is None else base_attr.clone()
        for edit_attr in edit_attr_list:
            if edit_attr > 0:
                new_attr[:, edit_attr - 1] += degree
            else:
                new_attr[:, -edit_attr - 1] -= degree
        attr_list.append(new_attr)

    @torch.no_grad()
    def generate_(self, num=0, num_image_tiles=5, type='mix'):
        self.GAN.eval()

        if 'mix' == type:
            return self.generate_mix(num, num_image_tiles)
        elif 'regular' == type:
            return self.generate_regular(num, num_image_tiles)
        elif 'age_moving' == type:
            return self.generate_age(num, num_image_tiles)
        elif 'random' == type:
            return self.generate_random(num)
            # return self.generate_random_cls(num)
        elif 'selected' == type:
            return self.generate_selected(num)

    def generate_mix(self, num, num_image_tiles):
        n_attr = len(self.selected_idx)
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '/' + 'generated'
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        if not dir_full.exists():
            os.mkdir(dir_full)

        # continuing attribute
        noi = image_noise(1, self.image_size, device=self.rank)

        attr = torch.ones(1, n_attr).cuda(self.rank) * 0.5

        # attr[:, [20, 39]] = 0.8
        # attr_list = [attr]

        # wqh 注释掉的 start
        # self.add_attr(attr_list, [-40], degree=0.6)
        # self.add_attr(attr_list, [16])
        # self.add_attr(attr_list, [1,17,23,-25])
        # self.add_attr(attr_list, [14])
        # self.add_attr(attr_list, [32])

        # self.add_attr(attr_list, [40], degree=0.6)
        # self.add_attr(attr_list, [-16])
        # self.add_attr(attr_list, [-1,-17,-23,25])
        # self.add_attr(attr_list, [-14])
        # self.add_attr(attr_list, [-32])
        # wqh 注释掉的 end

        attr[:, [39]] = 0.8
        attr[:, [20]] = 0.2
        attr_list = [attr]

        count_max_40 = 0
        for i in range(495):

            if count_max_40 == 40:
                count_max_40 = 0
            count_max_40 += 1

            if i < 100:
                self.add_attr(attr_list, [10 - count_max_40], degree=0.1)

            elif i >= 100 and i < 200:
                self.add_attr(attr_list, [20 - count_max_40], degree=0.1)

            elif i >= 200 and i < 300:
                self.add_attr(attr_list, [30 - count_max_40], degree=0.1)

            elif i >= 300 and i < 400:
                self.add_attr(attr_list, [40 - count_max_40], degree=0.1)

            elif i >= 400:
                self.add_attr(attr_list, [-count_max_40], degree=0.1)

            else:
                self.add_attr(attr_list, [count_max_40], degree=0.1)

        # self.add_attr(attr_list, [5], degree=0.6)
        # self.add_attr(attr_list, [-6])
        # self.add_attr(attr_list, [-7, -8])
        # self.add_attr(attr_list, [-9])
        # self.add_attr(attr_list, [-10])

        # smiling
        # new_attr = attr.clone()
        # new_attr[31] += 0.4
        # attr_list.append(new_attr)
        #
        # # age
        # new_attr = attr.clone()
        # new_attr[39] -= 0.4
        # attr_list.append(new_attr)
        #
        # # thin
        # new_attr = attr.clone()
        # new_attr[13] -= 0.4
        # attr_list.append(new_attr)

        # # white hair
        # new_attr = attr.clone()
        # new_attr[[8, 9]] -= 0.3
        # new_attr[17] += 0.3
        # attr_list.append(new_attr)
        #
        # # eyeglasses
        # new_attr = attr.clone()
        # new_attr[15] += 0.3
        # attr_list.append(new_attr)
        #
        # # beard
        # # new_attr = attr.clone()
        # # new_attr[[0,16,22]] += 0.3
        # # new_attr[24] -= 0.3
        # # attr_list.append(new_attr)
        # # makeup
        # new_attr = attr.clone()
        # new_attr[[18, 36]] += 0.4
        # attr_list.append(new_attr)

        # composite attribte for user study
        # new_attr = attr_list[-1].clone()
        # new_attr[:, 39] -= 0.3
        # new_attr[:, 8] -= 0.3
        # new_attr[:, 9] -= 0.3
        # new_attr[:, 17] += 0.3
        # attr_list.append(new_attr)
        #
        # new_attr = attr_list[-1].clone()
        # new_attr[:, [12, 31, 7]] += 0.3
        # attr_list.append(new_attr)
        #
        # new_attr = attr_list[-1].clone()
        # new_attr[:, [13, 14, 25]] += 0.3
        # attr_list.append(new_attr)

        moving_attrs = torch.stack(attr_list, dim=0)
        num_cols = len(moving_attrs)  # 24
        print("number of images", num_cols)
        for i in tqdm(range(num_image_tiles),
                      desc='Saving generated continuing attribute images'):  # num_image_tiles 生成图片数量
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            # latents[0][0] = latents[0][0][:1].expand(num_cols * n_attr, -1)
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # [1, 7, 512]
            w_styles = w_styles.expand(num_cols, -1, -1)
            b, l, _ = w_styles.shape
            wattrs = moving_attrs.expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)

            self.batch_size = num_cols
            # print(self.batch_size)
            generated_images = evaluate_in_chunks(self.batch_size, self.GAN.GE, w_styles, noi).clamp_(0., 1.)

            # print(generated_images.shape)
            result_dir = str(self.results_dir / 'random_everyone')

            # wqh add
            # path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-mix.{ext}')
            # print(generated_images.shape)

            # wqh
            for j in range(num_cols):
                path = result_dir + '/' + f'{str(i).zfill(4)}-{str(j).zfill(2)}{ext}'
                # print(generated_images[j].shape, path)
                torchvision.utils.save_image(generated_images[j], path)

            # torchvision.utils.save_image(generated_images, path, nrow=num_cols)
        with open(str(self.results_dir / dir_name / f'{str(num)}-mixing.txt'), 'w') as fout:
            fout.write(moving_attrs.__repr__())

    def generate_regular(self, num, num_image_tiles):
        n_attr = len(self.selected_idx)
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '/' + 'generated'
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        noi = image_noise(1, self.image_size, device=self.rank)

        num_cols, num_rows = 11, n_attr
        noi = noi.expand(num_cols * n_attr, -1, -1, -1)
        base_attr = torch.ones(n_attr).cuda(self.rank) * 0.5
        attr_list = []
        step = 1 / (num_cols - 1)
        for i in range(n_attr):
            for j in range(num_cols):
                new_attr = base_attr.detach().clone()
                new_attr[i] = j * step
                attr_list.append(new_attr)
        moving_attrs = torch.stack(attr_list, dim=0)

        for i in tqdm(range(num_image_tiles), desc='Saving generated continuing attribute images'):
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            # latents[0][0] = latents[0][0][:1].expand(num_cols * n_attr, -1)
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # [1, 7, 512]
            w_styles = w_styles.expand(num_cols * n_attr, -1, -1)
            b, l, _ = w_styles.shape
            wattrs = moving_attrs[:, None].expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)

            generated_images = evaluate_in_chunks(self.batch_size, self.GAN.GE, w_styles, noi).clamp_(0., 1.)

            path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-regular.{ext}')
            torchvision.utils.save_image(generated_images, path, nrow=num_cols)
        with open(str(self.results_dir / dir_name / f'{str(num)}-moving.txt'), 'w') as fout:
            fout.write(moving_attrs.__repr__())

        return dir_full

    def generate_age(self, num, num_image_tiles):
        n_attr = len(self.selected_idx)
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '/' + 'generated'
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        noi = image_noise(1, self.image_size, device=self.rank)

        result_dir = str(self.results_dir / 'age_moving')
        os.makedirs(result_dir, exist_ok=True)
        num_cols = 11  # 以前11
        attrs = torch.ones(num_cols, 1, n_attr) * 0.6  # 以前0.5
        attrs[:, 0, -1] = torch.linspace(0, 1, num_cols)  # n从0到1一共是11个值。1个属性向量40维，年龄从年轻到老，1个人11种属性变化，就得有11个40维的向量。
        moving_attrs = attrs.cuda(self.rank)

        for i in tqdm(range(num_image_tiles), desc='Saving generated continuing attribute images'):
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # [1, 7, 512]
            w_styles = w_styles.expand(num_cols, -1, -1)
            b, l, _ = w_styles.shape
            wattrs = moving_attrs.expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)  # w_styles相当于z，拼接了40维属性值，z决定了是谁，wattrs代表属性。

            generated_images = evaluate_in_chunks(self.batch_size, self.GAN.GE, w_styles, noi).clamp_(0., 1.)
            for j in range(num_cols):
                path = result_dir + '/' + f'{str(i).zfill(4)}-{str(j).zfill(2)}{ext}'
                # print(generated_images[j].shape, path)
                torchvision.utils.save_image(generated_images[j], path)

        return result_dir

    # 随机采集属性，用于计算FID
    def generate_random(self, num):
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '/random'
        result_dir = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        os.makedirs(result_dir, exist_ok=True)
        for i in tqdm(range(num), desc='Generating {} images with random sampled attributes'.format(num)):
            # 随机采样noise
            noi = image_noise(1, self.image_size, device=self.rank)
            # 随机采样attribute code
            attrs = torch.rand(1, 1, self.n_attr).cuda(self.rank)
            # 随机采样content code，即z，有两种采样方式
            # regular：z全部相同，我们在这里用这种
            # mix：z有两种，即style mixing
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            # z经过stylegan的线性层
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # 完成7个w[1,512]的拼接，[1, 7, 512]
            # 将采样的attrs拼接在w_styles后面
            b, l, _ = w_styles.shape
            wattrs = attrs.expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)
            # 送入ACGAN生成图片
            generated_images = self.GAN.GE(w_styles, noi).clamp_(0., 1.)
            path = result_dir / f'{str(i).zfill(6)}{ext}'
            torchvision.utils.save_image(generated_images[0], path)

        return result_dir

    def generate_random_cls(self, num):
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '_random'
        result_dir = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        os.makedirs(result_dir, exist_ok=True)
        correct_pred = torch.zeros(1, 40).cuda()
        total_pred = torch.ones(1, 40).cuda() * num
        degree = 2.0
        for i in tqdm(range(num), desc='Generating {} images with random sampled attributes'.format(num)):
            # 随机采样noise
            noi = image_noise(1, self.image_size, device=self.rank)
            # 随机采样attribute code
            attrs = torch.rand(1, 1, self.n_attr).cuda(self.rank)
            # 随机采样content code，即z，有两种采样方式
            # regular：z全部相同，我们在这里用这种
            # mix：z有两种，即style mixing
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            # z经过stylegan的线性层
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # 完成7个w[1,512]的拼接，[1, 7, 512]
            # 将采样的attrs拼接在w_styles后面
            b, l, _ = w_styles.shape
            wattrs = attrs.expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)
            # 送入ACGAN生成图片
            generated_images = self.GAN.GE(w_styles, noi).clamp_(0., 1.)
            classification_score = sigmoid(classfier(generated_images))
            path = result_dir / f'{str(i).zfill(6) + "_1"}{ext}'
            torchvision.utils.save_image(generated_images[0], path)

            factors = torch.where(classification_score <= 0.5, degree, 0 - degree)
            factors[0, :-1] = 0
            attrs2 = attrs + factors
            w_styles = styles_def_to_tensor(w_space)
            # 将采样的attrs拼接在w_styles后面
            b, l, _ = w_styles.shape
            wattrs2 = attrs2.expand(b, l, -1)
            w_styles2 = torch.cat([w_styles, wattrs2], dim=-1)
            # 送入ACGAN生成图片
            generated_images2 = self.GAN.GE(w_styles2, noi).clamp_(0., 1.)
            classification_score2 = sigmoid(classfier(generated_images2))
            print(i, classification_score[0, -1], classification_score2[0, -1])
            factors2 = torch.where(classification_score2 <= 0.5, degree, 0 - degree)
            factors2[0, :-1] = 0
            correct_pred += factors != factors2
            path = result_dir / f'{str(i).zfill(6) + "_2"}{ext}'
            torchvision.utils.save_image(generated_images2[0], path)

        avarage = (correct_pred / total_pred)[0]
        print(avarage.cpu().numpy())
        print(torch.mean(avarage).cpu().numpy())

        return result_dir

    # 生成指定属性的多张图片
    def generate_selected(self, num):
        latent_dim = self.GAN.G.latent_dim
        dir_name = self.name + '/selected'
        result_dir = Path().absolute() / self.results_dir / dir_name
        ext = '.jpg'

        os.makedirs(result_dir, exist_ok=True)
        for i in tqdm(range(num), desc='Generating {} images with random sampled attributes'.format(num)):
            # 随机采样noise
            noi = image_noise(1, self.image_size, device=self.rank)
            # 指定attrs
            attrs = torch.rand(1, 1, self.n_attr).cuda(self.rank)
            attrs[:, :, [8,9,11,17]] = 0.9
            # 随机采样content code，即z，有两种采样方式
            # regular：z全部相同，我们在这里用这种
            # mix：z有两种，即style mixing
            latents = noise_list(1, 7, latent_dim, device=self.rank)
            # z经过stylegan的线性层
            w_space = latent_to_w(self.GAN.SE, latents)
            w_styles = styles_def_to_tensor(w_space)  # 完成7个w[1,512]的拼接，[1, 7, 512]
            # 将采样的attrs拼接在w_styles后面
            b, l, _ = w_styles.shape
            wattrs = attrs.expand(b, l, -1)
            w_styles = torch.cat([w_styles, wattrs], dim=-1)
            # 送入ACGAN生成图片
            generated_images = self.GAN.GE(w_styles, noi).clamp_(0., 1.)
            path = result_dir / f'{str(i).zfill(6)}{ext}'
            torchvision.utils.save_image(generated_images[0], path)

        return result_dir

    # wqh add 
    def save_img(image_tensor, filename):
        image_numpy = image_tensor.float().numpy()
        image_numpy = image_numpy[:, 3:3 + 250, 28:28 + 200]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        image_pil = Image.fromarray(image_numpy)
        # image_pil = image_pil.resize((200, 250),Image.BILINEAR) 
        image_pil.save(filename, quality=95)
        print("Image saved as {}".format(filename))

    def test_encode(self):
        from stylegan2_pytorch.encode_image import encode_image

        for img_path in glob.glob('../data/CombinedFace/FFHQ1024/*.png'):
            img_name = os.path.basename(img_path)
            w_styles, attr, noi = encode_image(img_path, self.regressor, self.GAN.GE, resolution=256)

            ws_path = os.path.join('encoding', f'{img_name}_ws.pth')
            attr_path = os.path.join('encoding', f'{img_name}_attr.pth')
            noi_path = os.path.join('encoding', f'{img_name}_noi.pth')
            torch.save(w_styles, ws_path)
            torch.save(attr, attr_path)
            torch.save(noi, noi_path)

    def edit(self, img_path, attribute_processor):
        img_name = os.path.basename(img_path).split(".")[0]
        dir_name = self.name + '/' + 'single_edited'
        ext = self.image_extension

        dir_full = Path().absolute() / self.results_dir / dir_name
        if not dir_full.exists():
            os.mkdir(dir_full)

        # GAN inversion
        from stylegan2_pytorch.encode_image import encode_image
        ws_path = os.path.join('encoding', f'{img_name}_ws.pth')
        attr_path = os.path.join('encoding', f'{img_name}_attr.pth')
        noi_path = os.path.join('encoding', f'{img_name}_noi.pth')
        if os.path.exists(ws_path) and os.path.exists(noi_path):
            w_styles = torch.load(ws_path).cuda()
            attr = torch.load(attr_path).cuda()
            noi = torch.load(noi_path).cuda()
        else:
            w_styles, attr, noi, loss = encode_image(img_path, self.regressor, self.GAN.GE, resolution=256)
            # if loss > 2:
            #     return ''
            torch.save(w_styles, ws_path)
            torch.save(attr, attr_path)
            torch.save(noi, noi_path)

        moving_attrs = torch.cat(attribute_processor(attr), dim=0)

        # noi = noi.expand(len(moving_attrs), -1, -1, -1)
        # latents = [(latents, 7)]
        w_styles = w_styles.expand(len(moving_attrs), -1, -1)
        w_styles = self.cat_wstyle_attr(w_styles, moving_attrs)
        generated_images = self.GAN.GE(w_styles, noi).clamp_(0., 1.)

        for i in range(len(generated_images)):
            path = dir_full / f'{img_name}_{i}.{ext}'
            torchvision.utils.save_image(generated_images[i], path)

        return dir_full

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:],
                       duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid),
            ('Reg', self.gen_loss_reg)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.add_scalar(f'Loss/{name}', value, self.steps)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.config_path), True)
        rmtree(self.logdir)
        self.logger = SummaryWriter(f'{self.logdir}/{self.starttime}')
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])


class ModelLoader:
    def __init__(self, *, base_dir, name='default', load_from=-1):
        self.model = Trainer(name=name, base_dir=base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi=None):
        noise = noise.cuda()
        w = self.model.GAN.S(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.G.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device=0)

        images = self.model.GAN.G(w_tensors, noise)
        images.clamp_(0., 1.)
        return images
