import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from stylegan2_pytorch.regressor import build_regressor
from stylegan2_pytorch.dataset import Dataset, AttributeDataset, SceneryDataset
import torch


def test(dataroot):
    dataset = SceneryDataset(dataroot, 256)
    dataset.selected_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    dataloader = DataLoader(dataset, 1, num_workers=1)
    regressor = build_regressor(len(dataset.selected_idx))
    regressor = regressor.cuda()
    regressor.load_state_dict(torch.load('saves/scene/regressor_200.pth')["regressor"])
    regressor.eval()
    total_error = torch.zeros(len(dataset.selected_idx))
    with torch.no_grad():
        for img, attrs, _ in tqdm(dataloader):
            img = img.cuda()
            pred = regressor(img).cpu()
            # error = torch.abs(pred - attrs)
            error = (pred - attrs) ** 2
            total_error += error[0]
    print(total_error / len(dataset))

    names = ['dirty', 'daylight', 'night', 'sunrisesunset', 'dawndusk', 'sunny', 'clouds', 'fog', 'storm', 'snow', 'warm', 'cold', 'busy', 'beautiful', 'flowers', 'spring', 'summer', 'autumn', 'winter', 'glowing', 'colorful', 'dull', 'rugged', 'midday', 'dark', 'bright', 'dry', 'moist', 'windy', 'rain', 'ice', 'cluttered', 'soothing', 'stressful', 'exciting', 'sentimental', 'mysterious', 'boring', 'gloomy', 'lush'];
    l2_error = total_error / len(dataset);
    order = l2_error.argsort()
    for i in order:
        print(f'{names[i]}|{l2_error[i]}')


if __name__ == "__main__":
    # test('../../data/CombinedScene/')
    test('/home/jiaxuliu/GANs/lightweight-gan-z/results/scene256-5attrnew-random-w5/fixed')