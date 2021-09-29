import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataset import CelebADataset
from stylegan2_pytorch.regressor import build_regressor
import torch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter


class RegressorRunner:
    def __init__(self, regressor, optimizer, save_dir, name, epoch, phase='train', trainloader=None, testloader=None):
        self.regressor = regressor
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.save_dir = save_dir
        self.name = name
        self.epoch = epoch

        self.crirerion = nn.MSELoss()
        self.log_freq = 10
        self.save_freq = 1 if self.name == 'celeba' else 5
        self.test_freq = 50
        if phase == 'train':
            self.current_epoch = 1
            time = datetime.datetime.now().__format__('%Y-%m-%d %H:%M:%S')
            self.logger = SummaryWriter(f'runs/{self.name}_reg/{time}')

    def train(self, n_epoch=100):
        dataloader = self.trainloader
        self.regressor.train()
        global_step = 0
        for epoch in range(self.current_epoch, n_epoch + 1):
            step = 0
            running_loss = 0
            for data, attrs, img_names in tqdm(dataloader):
                step += 1
                global_step += 1
                data = data.cuda()
                attrs = attrs.cuda()
                output = self.regressor(data)

                self.optimizer.zero_grad()
                loss = self.crirerion(output, attrs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % self.log_freq == 0:
                    average_mse = running_loss / self.log_freq
                    print('Epoch {}, regressor MSE loss {:.4f}'.format(epoch, average_mse))
                    self.logger.add_scalar('mse', average_mse, global_step)
                    running_loss = 0

            if epoch % self.save_freq == 0:
                self.save(epoch)
            if epoch % self.test_freq == 0:
                self.test(epoch)

    @torch.no_grad()
    def test(self):
        num = self.current_epoch
        self.regressor.eval()
        dataloader = self.testloader
        norm_file_path = f'{self.save_dir}/{num}-attr.txt'  # path to save normed quantized attribute
        output_lines = []
        for data, attrs, img_names in tqdm(dataloader):
            data = data.cuda()
            output = self.regressor(data).detach().cpu().numpy()

            for img_name, pred_attrs in zip(img_names, output):
                line_list = [img_name]
                for pred_attr in pred_attrs:
                    line_list.append('{:.4f}'.format(pred_attr))
                output_lines.append(','.join(line_list) + '\n')
        with open(norm_file_path, 'w') as fout:
            for l in output_lines:
                fout.write(l)
        print()

    def save(self, epoch):
        print(f'saving epoch {epoch}')
        save_name = f'{self.save_dir}/regressor_{epoch}.pth'
        with open(f'{self.save_dir}/lastcheckpoint.txt', 'w') as fout:
            fout.write(save_name)
        save_data = {
            'regressor': self.regressor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(save_data, save_name)

    def load(self):
        lastcheckpoint_path = f'{self.save_dir}/lastcheckpoint.txt'
        if not os.path.exists(lastcheckpoint_path):
            print('checkpoint not found!')
            return
        else:
            with open(lastcheckpoint_path) as fin:
                checkpoint = fin.readline().strip()
        print(f'load checkpoint {checkpoint}')
        load_data = torch.load(checkpoint)

        self.regressor.load_state_dict(load_data['regressor'])
        self.optimizer.load_state_dict(load_data['optimizer'])
        self.current_epoch = load_data['epoch']


if __name__ == "__main__":

    save_dir = 'saves/CelebA'
    os.makedirs(save_dir, exist_ok=True)
    dataroot = './data/CelebA/'
    dataset = CelebADataset(dataroot, 256)
    trainloader = DataLoader(dataset, 64, num_workers=8)
    testloader = DataLoader(dataset, 64, num_workers=8, shuffle=False)

    regressor = build_regressor(len(dataset.selected_idx))
    regressor = regressor.cuda()
    reg_optimizer = optim.Adam(regressor.parameters(), lr=0.0001)
    runner = RegressorRunner(regressor, reg_optimizer, save_dir, args.dataset, trainloader, testloader)
    runner.train()
    runner.test()