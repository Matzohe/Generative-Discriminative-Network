from ModmaDataloader import MODMADataLoader as MMDL
from FFT_process import seperate_frequency_data
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm
from Config import set_seed

set_seed(42)


def wt_process(data, family, config_):
    """

    :param data: input data
    :param family: Wavelet families like db2,db6
    :return: two tensor, one is cA, another is cD
    """
    if not isinstance(family, str):
        raise TypeError("family should be a string")
    wt = pywt.Wavelet(family)
    data = seperate_frequency_data(data, config_["STFT_information"])

    if data.ndim == 1:
        cA, cD = pywt.dwt(data.numpy(), wavelet=wt)
        return cA, cD
    else:
        shape = np.array(data.shape).tolist()
        length = 0
        if isinstance(data, np.ndarray):
            new_data = torch.tensor(data).view(-1, data.shape[-1])
        else:
            new_data = data.view(-1, data.shape[-1])
        cA_list = []
        cD_list = []
        for each in range(new_data.shape[0]):
            cA, cD = pywt.dwt(new_data[each].numpy(), wavelet=wt)
            cA_list.append(cA)
            cD_list.append(cD)
            length = len(cA)
        shape[-1] = length
        new_cA = torch.from_numpy(np.concatenate(cA_list, axis=0)).view(shape)
        new_cD = torch.from_numpy(np.concatenate(cD_list, axis=0)).view(shape)
        return new_cA, new_cD


if __name__ == "__main__":
    root = r"G:\data package\MODMA_data\data_8trial.mat"
    depression_dataset = MMDL(root, 'depression')
    control_dataset = MMDL(root, 'control')
    data1 = depression_dataset[(0, 0)]
    data2 = control_dataset[(0, 0)]
    db6 = pywt.Wavelet('db6')
    a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15]).numpy()
    k = torch.randn(size=(12,))
    # 小波变换处理与重构
    cA, cD = pywt.dwt(data1[0].numpy(), 'db6')
    print(len(cA))
    reconstruct_data = pywt.idwt(cA, cD, 'db6')
    loss = F.mse_loss(torch.from_numpy(reconstruct_data), data1[0])
    print(data1.shape)
    loss_dict = []
    with tqdm(total=128) as pbar:
        for each in range(data1.shape[0]):
            cA, cD = pywt.dwt(data1[each].numpy(), 'db6')
            reconstruct_data = pywt.idwt(cA, cD, 'db6')
            # if each == 0:
            #     dA, dD = pywt.dwt(data1[each + 1].numpy(), 'db6')
            #     figure = plt.figure(figsize=(20, 24))
            #     plt.plot([i for i in range(len(dA))], cA - dA)
            #     plt.show()
            # 曼哈顿距离
            # if each == 0:
            #     figure = plt.figure(figsize=(20, 24))
            #     plt.plot([i for i in range(len(data1[each]))], data1[each])
            #     plt.plot([i for i in range(len(data1[each]))], reconstruct_data)
            #     plt.show()
            loss = torch.abs(torch.tensor(reconstruct_data) - data1[each]).sum()
            loss_dict.append(loss)
            pbar.update(1)
    # 这里查看还原的数据与原数据是否有很大的差距
    # plt.plot(torch.tensor(range(data1.shape[0])), loss_dict)
    # plt.show()
    # output, _ = wt_process(data1, 'db2', config)
    # print(output.shape)
