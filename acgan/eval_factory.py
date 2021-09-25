from tqdm import tqdm
import torch
from torch import nn
nn.MSELoss

class EvalTool:
    def execute(self, model, dataloader):
        pass


class ClassifierEval(EvalTool):

    def execute(self, model, dataloader, factor=0.05):
        model.eval()
        correct = 0
        total = 0
        sigmoid = nn.Sigmoid()
        for data, labels in tqdm(dataloader):
            data = data.cuda()
            output = model(data)
            output = sigmoid(output)
            output = output.detach().cpu()
            prediction = torch.where(output <= 0.5, -1, 1)
            correct += torch.sum(prediction == labels, dim=0).float()
            total += len(labels)

            if total >= factor * len(dataloader.dataset):
                break

        print(correct / total)
        print("Accuracy is {:.2f}".format(torch.mean(correct / total) * 100))


class RegressorEval(EvalTool):

    def execute(self, model, dataloader, factor=0.05):
        model.eval()
        mse = 0
        total = 0
        for data, labels in tqdm(dataloader):
            data = data.cuda()
            output = model(data)
            output = output.detach().cpu()
            mse += torch.square(output - labels).sum()
            total += len(labels)

            if total >= factor * len(dataloader.dataset):
                break

        print("MSE is {:.4f}".format(mse/total/40))

classifier_eval = ClassifierEval()
regressor_eval = RegressorEval()

def get(mode):
    if mode == 'classifier':
        return classifier_eval
    else:
        return regressor_eval
