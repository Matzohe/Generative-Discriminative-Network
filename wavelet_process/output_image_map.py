from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from Config import set_seed
from reconstruct_viewer import wavelet_reconstruct

set_seed(42)

# 这里是将所有输出绘制成一张图片，这张图片的大小为（128， 2500）
# 每个被试每段时间有两张图，第一张是自己类型的分类器对应的图片，其中，每张图片的值为预测的和原先的值的差的绝对值
# 第二张图则是结合两个分类器的差异进行绘制的，同样是根据差值的绝对值，但是变为两者差值的比例


def plot_new_image(original_signal, true_signal, false_signal):
    # 这里要求传入的信号必须是ndarray，不然会报错
    if not isinstance(original_signal, list):
        raise TypeError("输入的信号必须是包含128个channel的List")
    if not isinstance(original_signal[0], np.ndarray):
        raise TypeError("输入的信号必须是numpy.ndarray")
    data_list = []
    for i in range(len(original_signal)):
        new_data = np.abs(original_signal[i] - true_signal[i])
        new_data = (new_data - np.mean(new_data))
        data_list.append(torch.from_numpy(new_data))
    data_list = torch.softmax(torch.abs(torch.cat(data_list, dim=0).view(128, 2500)), dim=1).numpy() * 2500
    img = Image.fromarray(data_list)
    img.show()
    data_list_ = []
    for i in range(len(original_signal)):
        new_data_ = np.abs(original_signal[i] - false_signal[i])
        new_data_ = (new_data_ - np.mean(new_data_))
        data_list_.append(torch.from_numpy(new_data_))
    data_list_ = torch.softmax(torch.abs(torch.cat(data_list_, dim=0).view(128, 2500)), dim=1).numpy() * 2500
    img_ = Image.fromarray(data_list_)
    img_.show()
    # new_data_list = []
    # for j in range(len(original_signal)):
    #     new_data1 = np.abs(original_signal[j] - true_signal[j])
    #     new_data2 = np.abs(original_signal[j] - false_signal[j])
    #     data = new_data2 / new_data1
    #     data = (data - np.mean(data))
    #     new_data_list.append(torch.from_numpy(data))
    # new_data_list = torch.softmax(torch.cat(new_data_list, dim=0).view(128, 2500), dim=1).numpy()
    # new_img = Image.fromarray(new_data_list, 'P')
    # new_img.show()




if __name__ == '__main__':
    # 这里选取了一个depression病人的data进行打印，看看最后重建的脑电是否与真实脑电相近，如果相近，则这个理论是成立的
    original_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\original_save_wavelet_list.pt")
    depression_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\depression_save_wavelet_list.pt")
    control_data = torch.load(r"G:\python_program\brain_relate_net\middle_information\control_save_wavelet_list.pt")
    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    original_cA_data = torch.cat([each[0] for each in original_data['original_save_wavelet_list']], dim=0).view(-1,
                                                                                                                8,
                                                                                                                128,
                                                                                                                1255).cpu()
    original_cD_data = torch.cat([each[1] for each in original_data['original_save_wavelet_list']], dim=0).view(-1,
                                                                                                                8,
                                                                                                                128,
                                                                                                                1255).cpu()
    depression_cA_data = torch.cat([each[0] for each in depression_data['depression_save_wavelet_list']],
                                   dim=0).view(-1, 8, 128, 1255).cpu()
    depression_cD_data = torch.cat([each[1] for each in depression_data['depression_save_wavelet_list']],
                                   dim=0).view(-1, 8, 128, 1255).cpu()
    control_cA_data = torch.cat([each[0] for each in control_data['control_save_wavelet_list']], dim=0).view(-1, 8,
                                                                                                             128,
                                                                                                             1255).cpu()
    control_cD_data = torch.cat([each[1] for each in control_data['control_save_wavelet_list']], dim=0).view(-1, 8,
                                                                                                             128,
                                                                                                             1255).cpu()
    for i in range(32):
        for j in range(8):
            original_signal = wavelet_reconstruct('db6', original_cA_data[i][j], original_cD_data[i][j])
            depression_signal = wavelet_reconstruct('db6', depression_cA_data[i][j], depression_cD_data[i][j])
            control_signal = wavelet_reconstruct('db6', control_cA_data[i][j], control_cD_data[i][j])
            plot_new_image(original_signal, depression_signal, control_signal)
            break
        break

    original_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\original_control_wavelet_list.pt")
    depression_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\depression_control_save_wavelet_list.pt")
    # depression_data =\
    #     torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\depression_control_save_wavelet_list_200epoch.pt")
    control_data = torch.load(
        r"G:\python_program\brain_relate_net\middle_information\control_control_save_wavelet_list.pt")
    # control_data = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\middle_information\control_control_save_wavelet_list_200epoch.pt")
    # 原先保存的数据是以list的形式保存的，现在将其转化为numpy
    control_original_cA_data = torch.cat([each[0] for each in original_data['original_control_wavelet_list']],
                                         dim=0).view(-1, 8,
                                                     128,
                                                     1255).cpu()
    control_original_cD_data = torch.cat([each[1] for each in original_data['original_control_wavelet_list']],
                                         dim=0).view(-1, 8,
                                                     128,
                                                     1255).cpu()
    depression_control_cA_data = torch.cat(
        [each[0] for each in depression_data['depression_control_save_wavelet_list']], dim=0).view(
        -1, 8, 128, 1255).cpu()
    depression_control_cD_data = torch.cat(
        [each[1] for each in depression_data['depression_control_save_wavelet_list']], dim=0).view(
        -1, 8, 128, 1255).cpu()
    control_control_cA_data = torch.cat([each[0] for each in control_data['control_control_save_wavelet_list']],
                                        dim=0).view(-1, 8, 128,
                                                    1255).cpu()
    control_control_cD_data = torch.cat([each[1] for each in control_data['control_control_save_wavelet_list']],
                                        dim=0).view(-1, 8, 128,
                                                    1255).cpu()
    for i in range(32):
        for j in range(8):
            control_original_signal = wavelet_reconstruct('db6', control_original_cA_data[i][j], control_original_cD_data[i][j])
            control_control_signal = wavelet_reconstruct('db6', control_control_cA_data[i][j], control_control_cD_data[i][j])
            depression_control_signal = wavelet_reconstruct('db6', depression_control_cA_data[i][j],
                                                            depression_control_cD_data[i][j])
            plot_new_image(control_original_signal, control_control_signal, depression_control_signal)
            break
        break
