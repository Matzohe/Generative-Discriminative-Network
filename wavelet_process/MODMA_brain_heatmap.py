import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import OrderedDict
from torch.utils.data import Dataset
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from Config import ConfigSet, set_seed
from reconstruct_viewer import wavelet_reconstruct

set_seed(42)


def plot_explainable_information(config_, train_signal, test_signal, data_path, number):
    train_signal = torch.tensor(train_signal)
    test_signal = torch.tensor(test_signal)
    loss_function = nn.MSELoss()
    loss_list = []
    channel_position_information = np.loadtxt(config_['channel_position_root']) * 0.001
    for i in range(config_['channel_number']):
        loss = 1 / loss_function(train_signal[i], test_signal[i])
        loss_list.append(loss)
    data_info = mne.create_info(
        ch_names=["E%d" % (a + 1) for a in range(config_['channel_number'])],
        ch_types=['eeg' for _ in range(config_['channel_number'])],
        sfreq=100
    )
    channel_dict = {}
    for i in range(1, config_['channel_number'] + 1):
        channel_name = "E%d" % i
        channel_dict.update({channel_name: channel_position_information[i - 1]})
    montage = mne.channels.make_dig_montage(ch_pos=channel_dict)
    loss_list = torch.tensor(loss_list).view(-1, 1)
    evoked = mne.EvokedArray(loss_list, info=data_info)
    evoked.set_montage(montage)
    data_mean = torch.mean(loss_list)
    fig, axes = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap((evoked.data[:, 0] - data_mean.numpy()), evoked.info, axes=axes, show=False)
    plt.savefig(data_path + "/output_map{}.png".format(number))
    plt.close()

def depression_calculate_information(config_):
    depression_data_root = config_['control_depression_val_data_root']
    original_data_root = config_['original_depression_val_data_root']
    depression_list = torch.load(depression_data_root)['test_control_depression_save_wavelet_list_200epoch']
    depression_cA = torch.cat([data[0] for data in depression_list], dim=0).view(-1, 128, 1255)
    depression_cD = torch.cat([data[1] for data in depression_list], dim=0).view(-1, 128, 1255)
    original_list = torch.load(original_data_root)['test_original_depression_save_wavelet_list_200epoch']
    original_cA = torch.cat([data[0] for data in original_list], dim=0).view(-1, 128, 1255)
    original_cD = torch.cat([data[1] for data in original_list], dim=0).view(-1, 128, 1255)
    for i in range(depression_cA.shape[0]):
        train_signal = wavelet_reconstruct(config_['family'], depression_cA[i].cpu(), depression_cD[i].cpu())
        test_signal = wavelet_reconstruct(config_['family'], original_cA[i].cpu(), original_cD[i].cpu())
        plot_explainable_information(config_, train_signal, test_signal, config_['depression_img_save_root'], i)


def control_calculate_information(config_):
    control_data_root = config_['depression_control_val_data_root']
    original_data_root = config_['original_control_val_data_root']
    control_list = torch.load(control_data_root)['test_depression_control_save_wavelet_list_200epoch']
    control_cA = torch.cat([data[0] for data in control_list], dim=0).view(-1, 128, 1255)
    control_cD = torch.cat([data[1] for data in control_list], dim=0).view(-1, 128, 1255)
    original_list = torch.load(original_data_root)['test_original_control_wavelet_list_200epoch']
    original_cA = torch.cat([data[0] for data in original_list], dim=0).view(-1, 128, 1255)
    original_cD = torch.cat([data[1] for data in original_list], dim=0).view(-1, 128, 1255)
    for i in range(control_cA.shape[0]):
        train_signal = wavelet_reconstruct(config_['family'], control_cA[i].cpu(), control_cD[i].cpu())
        test_signal = wavelet_reconstruct(config_['family'], original_cA[i].cpu(), original_cD[i].cpu())
        plot_explainable_information(config_, train_signal, test_signal, config_['control_img_save_root'], i)


if __name__ == '__main__':
    config = ConfigSet(channel_number=128,
                       family='db6',
                       time_length=2500,
                       channel_position_root=r"G:\data package\MODMA_data\Electrod_information.txt",
                       depression_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\depression_save_wavelet_list_200epoch.pt",
                       control_depression_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\test_information\test_control_depression_save_wavelet_list_200epoch.pt",
                       original_depression_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\test_information\test_original_depression_save_wavelet_list_200epoch.pt",
                       control_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\control_control_save_wavelet_list_200epoch.pt",
                       depression_control_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\test_information\test_depression_control_save_wavelet_list_200epoch.pt",
                       original_control_val_data_root=r"G:\python_program\neuro_network_information\brain_relate_net\test_information\test_original_control_wavelet_list_200epoch.pt",
                       depression_img_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\brain_heatmap\10_channel\depression\c_test",
                       control_img_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\brain_heatmap\10_channel\control\d_test")
    depression_calculate_information(config)
    control_calculate_information(config)
