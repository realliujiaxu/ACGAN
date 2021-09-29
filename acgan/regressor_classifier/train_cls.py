import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from acgan.model_factory import ClsRegFactory
from dataset.dataset import CelebADatasetWithName
import torch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, model, optimizer, save_dir, name, trainloader=None, testloader=None):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.save_dir = save_dir
        self.name = name

        self.crirerion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.log_freq = 10
        self.save_freq = 1 if self.name == 'celeba' else 5
        self.test_freq = 50

        self.start_epoch = 1
        self.load()

        time = datetime.datetime.now().__format__('%Y-%m-%d %H:%M:%S')
        self.logger = SummaryWriter(f'runs/{self.name}_cls/{time}')

    def train(self, n_epoch=100):
        dataloader = self.trainloader
        self.model.train()
        global_step = 0
        for epoch in range(self.start_epoch, n_epoch+1):
            step = 0
            running_loss = 0
            for data, attrs, img_names in tqdm(dataloader):
                step += 1
                global_step += 1
                data = data.cuda()
                output = self.model(data)

                attrs = torch.clamp(attrs, min=0)
                attrs = attrs.type_as(output)
                attrs = attrs.cuda()

                self.optimizer.zero_grad()
                loss = self.crirerion(output, attrs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % self.log_freq == 0:
                    average_mse = running_loss / self.log_freq
                    print('Epoch {}, model CE loss {:.4f}'.format(epoch, average_mse))
                    self.logger.add_scalar('CE', average_mse, global_step)
                    running_loss = 0
            
            if epoch % self.save_freq == 0:
                self.save(epoch)
            if epoch % self.test_freq == 0:
                self.test(epoch)

    @torch.no_grad()
    def test(self, num = 0):
        self.model.eval()
        dataloader = self.testloader
        fout = open(f'{self.save_dir}/{num}-attr.txt', 'w')
        output_lines = []
        for data, attrs, img_names in tqdm(dataloader):
            data = data.cuda()
            output = self.model(data)
            # output = self.sigmoid(output)
            output = output.detach().cpu().numpy()

            for img_name, pred_attrs in zip(img_names, output):
                line_list = [img_name]
                for pred_attr in pred_attrs:
                    line_list.append('{:.4f}'.format(pred_attr))
                output_lines.append(','.join(line_list)+'\n')
        for l in output_lines:
            fout.write(l)
        fout.close()

    def save(self, epoch):
        print(f'saving epoch {epoch}')
        save_name = f'{self.save_dir}/model_{epoch}.pth'
        with open(f'{self.save_dir}/lastcheckpoint.txt', 'w') as fout:
            fout.write(save_name)
        save_data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(save_data, save_name)

    def load(self, epoch=-1):
        if epoch != -1:
            checkpoint = f'{self.save_dir}/model_{epoch}.pth'
        else:
            lastcheckpoint_path = f'{self.save_dir}/lastcheckpoint.txt'
            if not os.path.exists(lastcheckpoint_path):
                print('checkpoint not found!')
                return
            else:
                with open(lastcheckpoint_path) as fin:
                    checkpoint = fin.readline()
        print(f'continue from {checkpoint}')
        load_data = torch.load(checkpoint)

        self.model.load_state_dict(load_data['model'])
        self.optimizer.load_state_dict(load_data['optimizer'])
        self.start_epoch = load_data['epoch']

def train(dataroot, datasetclass, save_dir, name, n_epoch=300):
    dataset = datasetclass(dataroot, 256)
    trainloader = DataLoader(dataset, 32, num_workers=8)
    testloader = DataLoader(dataset, 32, num_workers=8, shuffle=False)

    model_factory = ClsRegFactory()
    model = model_factory.build("classifier")
    cls_optimizer = optim.Adam(model.parameters(), lr=0.001)

    runner = Runner(model, cls_optimizer, save_dir, name, trainloader, testloader)
    runner.train(n_epoch)
    runner.test(33)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Traing and testing attribute model.')
    parser.add_argument('--dataroot', default='../../data')
    parser.add_argument('--dataset', default='celeba', help="celeba")
    parser.add_argument('--save_dir', default='saves/', help="celeba")
    args = parser.parse_args()

    save_dir = args.save_dir + args.dataset + '_cls'
    os.makedirs(save_dir, exist_ok=True)
    train('../data/CombinedFace/celeba/', CelebADatasetWithName, save_dir, args.dataset, 100)