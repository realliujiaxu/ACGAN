import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from acgan.model_factory import TrainedClsRegFactory
from dataset.dataset import convert_transparent_to_rgb, expand_greyscale

regressor = TrainedClsRegFactory().get_scene_regressor()
regressor.eval()

image_size = 256
labels = ["night", "sunny", "clouds", "fog", "winter"]
label2base = {
    "night": 0.0592,
    "sunny": 0.2839,
    "clouds": 0.2317,
    "fog": 0.0692,
    "winter": 0.0307
}
for label in labels:
    dataset = ImageFolder("/home/jiaxuliu/GANs/result/lightweightgan-z/"+label,
                          transform=transforms.Compose(
                              [transforms.Lambda(convert_transparent_to_rgb),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(expand_greyscale(False))
                               ]
                          ))
    dataloader = DataLoader(dataset, 1, num_workers=1, shuffle=False)
    selected_idx = [2, 5, 6, 7, 18]

    positive_score = torch.zeros(1, 5)
    pos_count = 0
    negative_score = torch.zeros(1, 5)
    neg_count = 0
    with torch.no_grad():
        for img, y in tqdm(dataloader):
            score = regressor(img.cuda())[:, selected_idx]
            if y.item() == 0:
                positive_score += score.cpu()
                pos_count += 1
            else:
                negative_score += score.cpu()
                neg_count += 1

    # print(negative_score/neg_count, neg_count)
    print(label, positive_score/pos_count - label2base[label], pos_count)