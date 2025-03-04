import torch
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from wavelet_reconstruct import WaveletReconstruct
from reconstruct_viewer import wavelet_reconstruct
from Config import ConfigSet
import seaborn as sns


# 将depression和control的数据测试结果打印出来，并且逐渐增加对应channel number的情况绘制曲线图
# 先读入数据，计算准确率


def data_information_plot(loss_information, number, depression=True):
    # 输入的loss 应该是(40, 128)，从而能够计算最后的损失图像
    new_infor_list = []

    print(loss_information.shape[0])
    for i in range(loss_information.shape[0]):
        accurate_number = 0
        all_number = 0
        for j in range(loss_information.shape[-1]):
            if loss_information[i][j] < 0:
                accurate_number += 1
            all_number += 1
        if depression:
            new_infor_list.append(accurate_number)
        else:
            new_infor_list.append(128 - accurate_number)
    true_number = 0
    for i in range(number):
        if depression:
            if new_infor_list[i] >= 82:
                true_number += 1
        else:
            if new_infor_list[i] < 82:
                true_number += 1
    print("accurate rate:%f" % (true_number / number))
    new_infor_list = sorted(new_infor_list)
    accurate_list = []
    position_list = []
    for i in range(new_infor_list[0]):
        accurate_list.append(1.)
        position_list.append(i)
    for i in range(number - 1):
        for j in range(new_infor_list[i + 1] - new_infor_list[i]):
            accurate_list.append((number - 1 - i) / number)
            position_list.append(j + new_infor_list[i])
    plt.plot(position_list, accurate_list)
    plt.show()
    plt.close()
    sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    sns.distplot(new_infor_list, color="r", bins=30, kde=True)
    plt.show()
    return new_infor_list


if __name__ == '__main__':
    config = ConfigSet(
        map_number=8,
        channel_number=128,
        wavelet_length=1255,

        val_original_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_original_depression_save_wavelet_list_200epoch.pt",
        val_depression_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_depression_depression_save_wavelet_list_200epoch.pt",
        val_control_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_control_depression_save_wavelet_list_200epoch.pt",
        val_depression_number=4,
        val_original_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_original_control_wavelet_list_200epoch.pt",
        val_depression_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_depression_control_save_wavelet_list_200epoch.pt",
        val_control_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\val_control_control_save_wavelet_list_200epoch.pt",
        val_control_number=9,
        test_original_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_original_depression_save_wavelet_list_200epoch.pt",
        test_depression_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_depression_depression_save_wavelet_list_200epoch.pt",
        test_control_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_control_depression_save_wavelet_list_200epoch.pt",
        test_depression_number=5,
        test_original_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_original_control_wavelet_list_200epoch.pt",
        test_depression_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_depression_control_save_wavelet_list_200epoch.pt",
        test_control_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info\test_control_control_save_wavelet_list_200epoch.pt",
        test_control_number=5
    )
    original_data = torch.load(
        config['val_original_depression_data_root'])
    depression_data = torch.load(
        config['val_depression_depression_data_root'])
    control_data = torch.load(
        config['val_control_depression_data_root'])
    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    original_cA_data = torch.cat(
        [each[0] for each in original_data], dim=0).view(-1, 8,
                                                                                                       128,
                                                                                                       1255).cpu()
    original_cD_data = torch.cat(
        [each[1] for each in original_data], dim=0).view(-1, 8,
                                                                                                       128,
                                                                                                       1255).cpu()
    depression_cA_data = torch.cat(
        [each[0] for each in depression_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_cD_data = torch.cat(
        [each[1] for each in depression_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_cA_data = torch.cat(
        [each[0] for each in control_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_cD_data = torch.cat(
        [each[1] for each in control_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    # 在这里添加一些将生成的内容保存到指定文件夹中，之后对每个被试每个时段可以画一张地形图
    loss_function = nn.MSELoss()
    depression_loss = []
    for i in range(config['val_depression_number']):
        for j in range(config['map_number']):

            original_signal = wavelet_reconstruct('db6', original_cA_data[i][j], original_cD_data[i][j])
            depression_signal = wavelet_reconstruct('db6', depression_cA_data[i][j], depression_cD_data[i][j])
            control_signal = wavelet_reconstruct('db6', control_cA_data[i][j], control_cD_data[i][j])
            new_loss_list = []
            for k in range(config['channel_number']):
                loss_1 = loss_function(torch.tensor(original_signal[k]), torch.tensor(depression_signal[k])).sum()
                loss_2 = loss_function(torch.tensor(original_signal[k]), torch.tensor(control_signal[k])).sum()
                new_loss_list.append(loss_1 - loss_2)
            depression_loss.append(torch.tensor(new_loss_list))
    depression_loss = torch.cat(depression_loss, dim=0).view(
        config['val_depression_number'] * config['map_number'], config['channel_number'])
    depression_list = data_information_plot(depression_loss, config['val_depression_number'] * config['map_number'])

    original_data = torch.load(
        config['val_original_control_data_root'])
    depression_data = torch.load(
        config['val_depression_control_data_root'])

    control_data = torch.load(
        config['val_control_control_data_root'])

    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    control_original_cA_data = torch.cat(
        [each[0] for each in original_data],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_original_cD_data = torch.cat(
        [each[1] for each in original_data],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_control_cA_data = torch.cat(
        [each[0] for each in depression_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_control_cD_data = torch.cat(
        [each[1] for each in depression_data], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_control_cA_data = torch.cat(
        [each[0] for each in control_data],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_control_cD_data = torch.cat(
        [each[1] for each in control_data],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_loss = []

    for i in range(config['val_control_number']):
        for j in range(config['map_number']):
            control_original_signal = wavelet_reconstruct('db6', control_original_cA_data[i][j],
                                                          control_original_cD_data[i][j])
            control_control_signal = wavelet_reconstruct('db6', control_control_cA_data[i][j],
                                                         control_control_cD_data[i][j])
            depression_control_signal = wavelet_reconstruct('db6', depression_control_cA_data[i][j],
                                                            depression_control_cD_data[i][j])
            new_control_loss_list = []
            for k in range(config['channel_number']):
                loss_1 = loss_function(torch.tensor(control_original_signal[k]),
                                       torch.tensor(control_control_signal[k])).sum()
                loss_2 = loss_function(torch.tensor(control_original_signal[k]),
                                       torch.tensor(depression_control_signal[k])).sum()
                new_control_loss_list.append(loss_1 - loss_2)
            control_loss.append(torch.tensor(new_control_loss_list))

    control_loss = torch.cat(control_loss).\
        view(config['val_control_number'] * config['map_number'], config['channel_number'])

    control_list = data_information_plot(control_loss, config['val_control_number'] * config['map_number'],
                                         depression=False)
    sns.distplot(depression_list, color="r", bins=30, kde=True)
    sns.distplot(control_list, color="g", bins=30, kde=True)
    plt.show()
