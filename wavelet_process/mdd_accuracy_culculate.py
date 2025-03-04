import torch
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from wavelet_reconstruct import WaveletReconstruct
from reconstruct_viewer import wavelet_reconstruct
from accuracy_culculate import data_information_plot
from Config import ConfigSet
import seaborn as sns


if __name__ == '__main__':
    config = ConfigSet(
        map_number=14,
        batch_size=19,
        channel_number=19,
        wavelet_length=1285,

        val_original_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_original_depression_save_wavelet_list_200epoch.pt",
        val_depression_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_depression_depression_save_wavelet_list_200epoch.pt",
        val_control_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_control_depression_save_wavelet_list_200epoch.pt",
        val_original_depression_data_name='val_original_depression_save_wavelet_list_200epoch',
        val_depression_depression_data_name='val_depression_depression_save_wavelet_list_200epoch',
        val_control_depression_data_name='val_control_depression_save_wavelet_list_200epoch',
        val_depression_number=9,

        val_original_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_original_control_wavelet_list_200epoch.pt",
        val_control_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_control_control_save_wavelet_list_200epoch.pt",
        val_depression_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\val_depression_control_save_wavelet_list_200epoch.pt",
        val_original_control_data_name='val_original_control_wavelet_list_200epoch',
        val_depression_control_data_name='val_depression_control_save_wavelet_list_200epoch',
        val_control_control_data_name='val_control_control_save_wavelet_list_200epoch',
        val_control_number=5,

        test_original_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_original_depression_save_wavelet_list_200epoch.pt",
        test_depression_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_depression_depression_save_wavelet_list_200epoch.pt",
        test_control_depression_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_control_depression_save_wavelet_list_200epoch.pt",
        test_original_depression_data_name='test_original_depression_save_wavelet_list_200epoch',
        test_depression_depression_data_name='test_depression_depression_save_wavelet_list_200epoch',
        test_control_depression_data_name='test_control_depression_save_wavelet_list_200epoch',
        test_depression_number=5,

        test_original_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_original_control_wavelet_list_200epoch.pt",
        test_depression_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_depression_control_save_wavelet_list_200epoch.pt",
        test_control_control_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information\test_control_control_save_wavelet_list_200epoch.pt",
        test_original_control_data_name='test_original_control_wavelet_list_200epoch',
        test_depression_control_data_name='test_depression_control_save_wavelet_list_200epoch',
        test_control_control_data_name='test_control_control_save_wavelet_list_200epoch',
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
        [each[0] for each in original_data[config['val_original_depression_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    original_cD_data = torch.cat(
        [each[1] for each in original_data[config['val_original_depression_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_cA_data = torch.cat(
        [each[0] for each in depression_data[config['val_depression_depression_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_cD_data = torch.cat(
        [each[1] for each in depression_data[config['val_depression_depression_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_cA_data = torch.cat(
        [each[0] for each in control_data[config['val_control_depression_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_cD_data = torch.cat(
        [each[1] for each in control_data[config['val_control_depression_data_name']]], dim=0).view(
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
        [each[0] for each in original_data[config['val_original_control_data_name']]],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_original_cD_data = torch.cat(
        [each[1] for each in original_data[config['val_original_control_data_name']]],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_control_cA_data = torch.cat(
        [each[0] for each in depression_data[config['val_depression_control_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    depression_control_cD_data = torch.cat(
        [each[1] for each in depression_data[config['val_depression_control_data_name']]], dim=0).view(
        -1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_control_cA_data = torch.cat(
        [each[0] for each in control_data[config['val_control_control_data_name']]],
        dim=0).view(-1, config['map_number'], config['channel_number'], config['wavelet_length']).cpu()
    control_control_cD_data = torch.cat(
        [each[1] for each in control_data[config['val_control_control_data_name']]],
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

    control_loss = torch.cat(control_loss). \
        view(config['val_control_number'] * config['map_number'], config['channel_number'])

    control_list = data_information_plot(control_loss, config['val_control_number'] * config['map_number'],
                                         depression=False)
    sns.distplot(depression_list, color="r", bins=30, kde=True)
    sns.distplot(control_list, color="g", bins=30, kde=True)
    plt.show()