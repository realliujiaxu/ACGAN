import torch
from torch import nn

from acgan.model_factory import TrainedClsRegFactory
from acgan.tester import Tester
from acgan_config import face_args
from stylegan2_pytorch import Trainer
from stylegan2_pytorch.helper import add_attr, cast_list

sigmoid = nn.Sigmoid()


def load_model(args):
    model_args = dict(
        selected_idx = args.selected_idx,
        reg_weight = args.reg_weight,
        latent_dim = args.latent_dim,
        name = args.name,
        results_dir = args.results_dir,
        models_dir = args.models_dir,
        batch_size = args.batch_size,
        gradient_accumulate_every = args.gradient_accumulate_every,
        image_size = args.image_size,
        network_capacity = args.network_capacity,
        fmap_max = args.fmap_max,
        transparent = args.transparent,
        lr = args.learning_rate,
        lr_mlp = args.lr_mlp,
        ttur_mult = args.ttur_mult,
        rel_disc_loss = args.rel_disc_loss,
        num_workers = args.num_workers,
        save_every = args.save_every,
        evaluate_every = args.evaluate_every,
        num_image_tiles = args.num_image_tiles,
        trunc_psi = args.trunc_psi,
        fp16 = args.fp16,
        cl_reg = args.cl_reg,
        fq_layers = args.fq_layers,
        fq_dict_size = args.fq_dict_size,
        attn_layers = args.attn_layers,
        no_const = args.no_const,
        aug_prob = args.aug_prob,
        aug_types = cast_list(args.aug_types),
        top_k_training = args.top_k_training,
        generator_top_k_gamma = args.generator_top_k_gamma,
        generator_top_k_frac = args.generator_top_k_frac,
        dataset_aug_prob = args.dataset_aug_prob,
        calculate_fid_every = args.calculate_fid_every,
        mixed_prob = args.mixed_prob,
        log = args.log
    )

    trainer = Trainer(**model_args)
    trainer.load(args.load_from)

    return Tester(trainer.GAN, args.image_size)


def generate():
    tester = load_model(face_args)

    n_attr = len(face_args.selected_idx)
    attr = torch.ones(1, n_attr).cuda(0) * 0.5

    attr_list = []
    # TODO: 改属性值，注意这里的属性值要从1开始数
    add_attr(attr_list, [-40], attr, 0.3)
    add_attr(attr_list, [40], attr, 0.3)

    generated_images = tester.generate_images(attr_list)
    # torchvision.utils.save_image(generated_images[0], path)

    # 载入训练好的人脸属性分类器
    classifier = TrainedClsRegFactory().get_trained_classifier()

    classification_score = sigmoid(classifier(generated_images))

    print(classification_score[:, -1])


if __name__ == '__main__':
    generate()
