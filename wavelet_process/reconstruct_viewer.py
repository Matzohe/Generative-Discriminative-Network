import torch
import torch.nn as nn
import numpy as np
import pywt
import matplotlib.pyplot as plt
from Config import set_seed
import os

set_seed(42)
# 这个文档的作用是将传入的cA，cD重构回脑电，要求输入的数据为二维的,在测试时使用


def wavelet_reconstruct(family, cA, cD):
    if not cA.dim == 2 and cD.dim == 2:
        raise TypeError("在小波重构的过程中，传入的cA，cD必须要是二维的")
    if torch.is_tensor(cA):
        cA = cA.numpy()
        cD = cD.numpy()
    elif isinstance(cA, list):
        cA = np.array(cA)
        cD = np.array(cD)
    elif isinstance(cA, np.ndarray):
        pass
    else:
        raise TypeError("在小波重构时，传入的cA、cD必须是tensor、list或者np.array")
    signal_list = []
    for i in range(cA.shape[0]):
        new_signal = pywt.idwt(cA[i], cD[i], family)
        signal_list.append(new_signal)

    return signal_list

# 这个函数的作用是，将输入的cA，cD重构的信号打印并展示，其中，输入三组小波信号，一组是原信号，另外两组分别是depression分类器生成的
# 数据以及control分类器生成的数据，并且设置了打印颜色，原信号颜色为蓝色，而depression分类器生成的信号则为红色，control为绿色


def plt_signal(original_signal, depression_signal, control_signal, channel_number,
               save_path, signal_length=2500, color_list=None):
    if color_list is None:
        color_list = ['b', 'r', 'y']
    signal_list = [original_signal, depression_signal, control_signal]
    # fix, axes = plt.subplots(3, 1, figsize=(24, 24))
    # for i in range(3):
    #     ax = axes[i]
    #     ax.plot([j for j in range(signal_length)], signal_list[i][channel_number], color=color_list[i])
    # plt.show()
    for i_ in range(3):
        plt.plot([j_ for j_ in range(signal_length)], signal_list[i_][channel_number], color=color_list[i_])
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    # 这里选取了一个depression病人的data进行打印，看看最后重建的脑电是否与真实脑电相近，如果相近，则这个理论是成立的
    original_data = torch.load(r"G:\python_program\brain_relate_net\middle_information\original_save_wavelet_list.pt")
    depression_data = torch.load(r"G:\python_program\brain_relate_net\middle_information\depression_save_wavelet_list.pt")
    control_data = torch.load(r"G:\python_program\brain_relate_net\middle_information\control_save_wavelet_list.pt")
    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    original_cA_data = torch.cat([each[0] for each in original_data['original_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    original_cD_data = torch.cat([each[1] for each in original_data['original_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    depression_cA_data = torch.cat([each[0] for each in depression_data['depression_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    depression_cD_data = torch.cat([each[1] for each in depression_data['depression_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    control_cA_data = torch.cat([each[0] for each in control_data['control_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    control_cD_data = torch.cat([each[1] for each in control_data['control_save_wavelet_list']], dim=0).view(-1, 8, 128, 1255).cpu()
    # 在这里添加一些将生成的内容保存到指定文件夹中，之后对每个被试每个时段可以画一张地形图
    save_path = r"C:\Users\matzoh\Desktop\my project\wavelet based project\Modma_output_information\depression"
    for i in range(5):
        for j in range(8):

            original_signal = wavelet_reconstruct('db6', original_cA_data[i][j], original_cD_data[i][j])
            depression_signal = wavelet_reconstruct('db6', depression_cA_data[i][j], depression_cD_data[i][j])
            control_signal = wavelet_reconstruct('db6', control_cA_data[i][j], control_cD_data[i][j])
            for k in range(128):
                new_save_path = os.path.join(save_path, "patient{}_sequence{}_channel{}.png".format(i + 1, j + 1, k + 1))
                plt_signal(original_signal, depression_signal, control_signal,
                           save_path=new_save_path, channel_number=k)

    original_data = torch.load(r"G:\python_program\brain_relate_net\middle_information\original_control_wavelet_list.pt")
    depression_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\depression_control_save_wavelet_list.pt")
    # depression_data =\
    #     torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\depression_control_save_wavelet_list_200epoch.pt")
    control_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\control_control_save_wavelet_list.pt")
    # control_data = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\control_control_save_wavelet_list_200epoch.pt")
    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    control_original_cA_data = torch.cat([each[0] for each in original_data['original_control_wavelet_list']], dim=0).view(-1, 8,
                                                                                                                128,
                                                                                                                1255).cpu()
    control_original_cD_data = torch.cat([each[1] for each in original_data['original_control_wavelet_list']], dim=0).view(-1, 8,
                                                                                                                128,
                                                                                                                1255).cpu()
    depression_control_cA_data = torch.cat([each[0] for each in depression_data['depression_control_save_wavelet_list']], dim=0).view(
        -1, 8, 128, 1255).cpu()
    depression_control_cD_data = torch.cat([each[1] for each in depression_data['depression_control_save_wavelet_list']], dim=0).view(
        -1, 8, 128, 1255).cpu()
    control_control_cA_data = torch.cat([each[0] for each in control_data['control_control_save_wavelet_list']], dim=0).view(-1, 8, 128,
                                                                                                             1255).cpu()
    control_control_cD_data = torch.cat([each[1] for each in control_data['control_control_save_wavelet_list']], dim=0).view(-1, 8, 128,
                                                                                                             1255).cpu()
    save_path = r"C:\Users\matzoh\Desktop\my project\wavelet based project\Modma_output_information\control"
    for i in range(5):
        for j in range(8):
            control_original_signal = wavelet_reconstruct('db6', control_original_cA_data[i][j], control_original_cD_data[i][j])
            control_control_signal = wavelet_reconstruct('db6', control_control_cA_data[i][j], control_control_cD_data[i][j])
            depression_control_signal = wavelet_reconstruct('db6', depression_control_cA_data[i][j], depression_control_cD_data[i][j])
            for k in range(128):
                new_save_path = os.path.join(save_path,
                                             "control{}_sequence{}_channel{}.png".format(i + 1, j + 1, k + 1))
                plt_signal(control_original_signal, depression_control_signal, control_control_signal,
                           save_path=new_save_path, channel_number=k)
