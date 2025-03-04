import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Config import ConfigSet, set_seed
from collections import OrderedDict

from ftt_wavelet_dataloader import FftWaveletDataset

set_seed(42)


def data_process(data_input):
    # 将此时传入的数据转化为有利于cnn的情况，分为两组，一组是cA的数据，另一组是cD的数据
    # 传入的数据：([-1, batch_size, 11, 1251], [-1, batch_size, 11, 1251])
    # 此处数据为一组data的cA和cD信息
    # 注意此时正确的数据为data[0], 而剩下的均为训练的数据
    return data_input[0][:, :, 0, :], data_input[0][:, :, 1:, :], data_input[1][:, :, 0, :], data_input[1][:, :, 1:, :]


class ConvolutionNode(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size, stride, padding=0):
        super(ConvolutionNode, self).__init__()
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
        self.net = nn.Sequential(OrderedDict([('conv' + str(i), ConvolutionNode(config_['input_channel'][i],
                                                                                config_['hidden_channel'][i],
                                                                                config_['kernel_size'][i],
                                                                                config_['stride'][i]))
                                              for i in range(len(config_["input_channel"]))]))

    def forward(self, data_input):
        return self.net(data_input)


# 这里cnn处理完之后，将其结果处理完后计算损失, 这里使用反卷积，通过两个卷积层得到两个一维的tensor，之后经过concat，
# 目标tensor的形状为：[-1, batch_size, 1, 1255]
# 而上面网络的输入形状为: [-1, batch_size, 10, 1255]
# 注意，这里还需要谨慎检查一下之前数据处理的输出是否是正常的
# 之后通过全连接来将两者结合，对于cA和cD，如何将两者信息整合起来？通过train两个网络，分别对应预测的cA和cD


class ClassificationNode(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ClassificationNode, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, data_input):
        x = self.layer_norm(data_input)
        x = self.linear2(x) + x
        x = self.linear(x)
        return x


class ClassificationNet(nn.Module):
    def __init__(self, config_):
        super(ClassificationNet, self).__init__()
        self.config = config_
        self.concat_linear = nn.Linear(2, 1)
        self.cA_net = nn.Sequential(OrderedDict([('cA_linear' + str(i),
                                                  ClassificationNode(config_['input_linear_dim'][i],
                                                                     config_['hidden_linear_dim'][i]))
                                                 for i in range(len(config_['input_linear_dim']))]))

        self.cD_net = nn.Sequential(OrderedDict([('cD_linear' + str(i),
                                                  ClassificationNode(config_['input_linear_dim'][i],
                                                                     config_['hidden_linear_dim'][i]))
                                                 for i in range(len(config_['input_linear_dim']))]))

    def forward(self, data_input):
        # 此时输入形状应为（batch_size, 1， hidden_time_len）
        cA_data, cD_data = data_input
        new_data = torch.cat([cA_data.squeeze(1), cD_data.squeeze(1)], dim=-2).permute(0, 2, 1)
        new_data = self.concat_linear(new_data).permute(0, 2, 1).squeeze(1)
        # 这里修改了一下分析层的输出，包含了两个特征加上一个隐藏向量
        return self.cA_net(new_data), self.cD_net(new_data), new_data


if __name__ == '__main__':
    config = ConfigSet(conv_mean=0, conv_std=0.5, linear_mean=0, linear_std=0.5,
                       depression_train_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_depression_train_data.pt",
                       depression_val_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_depression_val_data.pt",
                       depression_test_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_depression_test_data.pt",
                       control_train_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_control_train_data.pt",
                       control_val_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_control_val_data.pt",
                       control_test_data_path=r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_control_test_data.pt",
                       batch_size=32,
                       epoch_number=15, lr=0.00005,
                       input_channel=[1, 4, 10, 6, 4, 2],
                       hidden_channel=[4, 10, 6, 4, 2, 1],
                       kernel_size=[(2, 4), (2, 4), (2, 4), (1, 4), (1, 4), (1, 4)],
                       stride=[(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
                       input_linear_dim=[300, 512, 1024, 1255],
                       hidden_linear_dim=[512, 1024, 1255, 1255])
    depression_dataset = FftWaveletDataset(config['depression_train_data_path'])
    cA, cD = depression_dataset[0]
    new_conv = ConvolutionNet(config)
    new_linear = ClassificationNet(config)
    with torch.no_grad():
        print(cA.shape)
        output = new_conv(cA[:, 1:, :].unsqueeze(1))
        output = new_linear((output, output))
        print(output[0].shape)
