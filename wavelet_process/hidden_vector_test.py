import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy
from Config import ConfigSet, set_seed

set_seed(42)


class NewDataset(Dataset):
    def __init__(self, config_, name):
        super(NewDataset, self).__init__()
        self.data_path = config_[name]
        self.new_data = torch.load(self.data_path)
        self.config_ = config_

    def __len__(self):
        return self.new_data.shape[0]

    def __getitem__(self, index):
        # 因为输入的数据是(128, 2, 300) 的形状
        return self.new_data[index].unsqueeze(0).to(self.config_['device'])


class CnnNode(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size, stride, padding=0):
        super(CnnNode, self).__init__()
        self.convolution = nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        self.convolution2 = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1)
        self.batch_normalization = nn.BatchNorm2d(input_channel)

    def forward(self, data_input):
        data_input = self.batch_normalization(data_input)
        data_input = data_input + self.convolution2(data_input)
        return self.convolution(data_input)


class ConvolutionNet(nn.Module):
    def __init__(self, config_):
        super(ConvolutionNet, self).__init__()
        self.config_ = config_
        self.net = nn.Sequential(OrderedDict([('conv' + str(i), CnnNode(config_['input_channel'][i],
                                                                        config_['hidden_channel'][i],
                                                                        config_['kernel_size'][i],
                                                                        config_['stride'][i]))
                                              for i in range(len(config_["input_channel"]))]))

    def forward(self, data_input):
        return self.net(data_input)


class LinearNode(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinearNode, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, data_input):
        x = self.layer_norm(data_input)
        x = self.linear2(x) + x
        x = self.linear(x)
        return x


def train(config_, depression_dataset, control_dataset, depression_dataset2, control_dataset2):
    cnn_network = ConvolutionNet(config_).to(device=config_['device'])
    linear_network = nn.Sequential(OrderedDict([('linear' + str(i),
                                                 LinearNode(config_['input_linear_dim'][i],
                                                            config_['hidden_linear_dim'][i]))
                                                for i in range(len(config_['input_linear_dim']))])).to(
                                                                                            device=config_['device'])
    optimizer = optim.Adam([{'params': cnn_network.parameters()},
                            {'params': linear_network.parameters()}], lr=config_['lr'])
    loss_function = nn.CrossEntropyLoss().to(device=config_['device'])
    depression_loss_list = []
    control_loss_list = []
    with tqdm(total=config_['epoch_num']) as pbar:

        for i in range(config_['epoch_num']):
            for j in range(len(depression_dataset)):
                optimizer.zero_grad()
                data_input = depression_dataset[j]
                hidden = cnn_network(data_input).view(data_input.shape[0], -1)
                output = torch.softmax(linear_network(hidden), dim=-1)
                loss1 = loss_function(output, torch.tensor((1,), device=config_['device']))
                loss1.requires_grad_(True)

                depression_loss_list.append(loss1.sum().detach().cpu())

                data_input = control_dataset[j]
                hidden = cnn_network(data_input).view(data_input.shape[0], -1)
                output = torch.softmax(linear_network(hidden), dim=-1)
                loss2 = loss_function(output, torch.tensor((0,), device=config_['device']))
                loss2.requires_grad_(True)
                loss = loss1.sum() + loss2.sum()
                loss.backward()
                control_loss_list.append(loss2.sum().detach().cpu())
                optimizer.step()
            pbar.update(1)
        plt.plot([i for i in range(config_['epoch_num'] * len(depression_dataset))], depression_loss_list)
        plt.show()
        plt.close()
        plt.plot([i  for i in range(len(control_loss_list))], control_loss_list)
        plt.show()
        plt.close()

    depression_accurate_list = []
    depression_accurate_number = 0
    depression_all_number = 0
    control_accurate_list = []
    control_accurate_number = 0
    control_all_number = 0
    with torch.no_grad():

        for j in range(len(depression_dataset2)):
            depression_all_number += 1
            optimizer.zero_grad()
            data_input = depression_dataset2[j]
            hidden = cnn_network(data_input).view(data_input.shape[0], -1)
            output = torch.softmax(linear_network(hidden), dim=-1)
            if output.argmax(dim=-1) == 1:
                depression_accurate_number += 1
                depression_accurate_list.append(j)

        for j in range(len(control_dataset2)):
            control_all_number += 1
            optimizer.zero_grad()
            data_input = control_dataset2[j]
            hidden = cnn_network(data_input).view(data_input.shape[0], -1)
            output = torch.softmax(linear_network(hidden), dim=-1)
            if output.argmax(dim=-1) == 0:
                control_accurate_number += 1
                control_accurate_list.append(j)
    print("depression_accurate_number:", depression_accurate_number)
    print("depression_all_number", depression_all_number)
    print("control_accurate_number", control_accurate_number)
    print("control_all_number", control_all_number)


if __name__ == '__main__':
    config = ConfigSet(input_channel=[128, 64, 32, 16],
                       hidden_channel=[64, 32, 16, 8],
                       kernel_size=[(2, 2), (1, 2), (1, 2), (1, 2)],
                       stride=[2, 2, 2, 2],
                       input_linear_dim=[144, 32, 4],
                       hidden_linear_dim=[32, 4, 2],
                       depression_val=r"G:\python_program\neuro_network_information\brain_relate_net\another_information\depression\val.pt",
                       depression_test=r"G:\python_program\neuro_network_information\brain_relate_net\another_information\depression\test.pt",
                       control_val=r"G:\python_program\neuro_network_information\brain_relate_net\another_information\control\val.pt",
                       control_test=r"G:\python_program\neuro_network_information\brain_relate_net\another_information\control\test.pt",
                       epoch_num=20,
                       device='cuda',
                       lr=0.0005)
    depression_val_dataset = NewDataset(config, "depression_val")
    depression_test_dataset = NewDataset(config, 'depression_test')
    control_val_dataset = NewDataset(config, 'control_val')
    control_test_dataset = NewDataset(config, 'control_test')
    train(config, depression_test_dataset, control_test_dataset, depression_val_dataset, control_val_dataset)
