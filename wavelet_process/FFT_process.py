import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from MddDataloader import MddDataLoader
from ModmaDataloader import MODMADataLoader as MMDT

# 这个cell写的是一个分离脑电不同频率成分的函数
# 分离不同成分频率之后，加上初始相位后重新拼接为不同频段的脑电


def FrequencyDomainSeperate(input_data, STFT_information):
    """
    STFT_information 为一个字典，其中包含这么几个元素
    time_len: 这个值在scipy.fftpack中是默认为信号总长度的,即与香农采样定理相匹配
    fstart: 我们想要提取不同频率的信号，这是每一band的开始频率
    fend: 这是每一band的截止频率
    foriginal: 这是原始采集数据的频率
    window: 这是STFT的窗口长度（秒）, 这里有window * foriginal = time_len

    输入数据的形状为（channel_number, time_sequence）,类型为tensor，中间注意数据类型转化为array，不然fft好像无法工作
    返回的数据为一个list,里面存放了len（fstart）组数据，每组数据中有channel_number个frequency
    """
    fstart = STFT_information['fstart']
    fend = STFT_information['fend']
    time_len = STFT_information['time_len']
    foriginal = STFT_information['foriginal']

    # 由于最后fft函数返回的是（0 ~ foriginal/2）的频率，并将这么多的频率分为time_len / 2份
    fstart_list = np.zeros(len(fstart), dtype=int)
    fend_list = np.zeros(len(fend), dtype=int)
    if len(fstart) != len(fend):
        raise RuntimeError("信号区间需要一个开始频率与一个截止频率，而这两者的列表长度不同")

    # 这一步是输出我们想要的信号在dft中输出的位置（index）
    for i in range(len(fstart)):
        fstart_list[i] = (fstart[i] - 1) * time_len // foriginal
        fend_list[i] = fend[i] * time_len // foriginal

    # 接下去是生成stft所需要的窗口函数，这里使用的是余弦窗口
    x = np.linspace(-1, 1, time_len)
    y = 0.55 + 0.5 * np.cos(x * np.pi)
    stft_data = input_data * y
    stft_output = fft(stft_data.numpy())
    # 最后返回fft操作后振幅的模值，还有相位可以返回，但这里并没有处理
    stft_frequency = torch.tensor(np.abs(stft_output))
    stft_angle = torch.tensor(np.angle(stft_output))
    frequency_list = []
    angle_list = []
    for i in range(len(fstart)):
        frequency_list.append(stft_frequency[:, fstart_list[i]: fend_list[i]])
        angle_list.append(stft_angle[:, fstart_list[i]: fend_list[i]])
    return frequency_list, angle_list


def seperate_frequency_data(input_data, STFT_information):
    # 接下去写一个函数，这个函数将脑电信号约束在14HZ以内，之后以这个数据进行小波变换
    channel_frequency, channel_angle = FrequencyDomainSeperate(input_data, STFT_information)
    ax_number = len(STFT_information['fstart'][1: 3])
    # fig, axes = plt.subplots(5, 1, figsize=(20, 250))
    names = []
    all_signal = []
    for name, _ in matplotlib.colors.cnames.items():
        names.append(name)
    for _i in range(ax_number):
        frequency_number = channel_frequency[_i].shape[-1]
        all_sum = 1
        synthesis_signal = torch.zeros([channel_frequency[_i].shape[-2], STFT_information['time_len']])
        for j in range(frequency_number):
            this_frequency = STFT_information['fstart'][_i] - 1 + j * STFT_information['foriginal'] / STFT_information['time_len']
            this_signal = channel_frequency[_i][:, j].reshape(-1, 1) * torch.cos(torch.arange(0, STFT_information['time_len']).reshape(1, -1) / STFT_information['foriginal'] * 2 * np.pi * this_frequency + channel_angle[_i][:, j].reshape(-1, 1))
            synthesis_signal += this_signal / STFT_information['time_len'] * STFT_information['window']
        x = np.linspace(-1, 1, STFT_information['time_len'])
        y = 0.55 + 0.5 * np.cos(x * np.pi)
        y = torch.tensor(y)
        synthesis_signal /= y
        # ax = axes[_i]
        all_signal.append(synthesis_signal)
        # 这里的形状为(128, time_len)

        # for k in range(128):
        #     ax.plot(torch.arange(STFT_information['time_len']), synthesis_signal[k], color=names[k])
    # plt.show()

    # return all_signal
    _new_signal = all_signal[0]
    for _i in range(len(all_signal) - 1):
        _new_signal += all_signal[_i + 1]
    return _new_signal


if __name__ == "__main__":
    data_path = r"G:\data package\MDD_data\mdd_pt"
    depression_dataset = MddDataLoader(data_path, "depression", device='cpu')
    control_dataset = MddDataLoader(data_path, "control", device='cpu')
    modma_dataloader = MMDT(r"G:\data package\MODMA_data\data_8trial.mat", 'depression')
    depression_data1 = modma_dataloader[(0, 0)]
    # depression_data1 = depression_dataset[0][:19, :2560]
    STFT_information = {'time_len': 2500,
                        'fstart': [1, 4, 8, 14, 31],
                        'fend': [3, 7, 13, 30, 50],
                        'foriginal': 250,
                        'window': 10}
    new_signal = seperate_frequency_data(depression_data1, STFT_information)
    print(new_signal.shape)
    plt.plot([i for i in range(2500)], new_signal[0])
    plt.show()
    # new_add_signal = new_signal[0]
    # for i in range(len(new_signal) - 1):
    #     new_add_signal += new_signal[i + 1]
    # loss = new_add_signal - depression_data1
    # fig = plt.figure(figsize=(20, 240))
    # plt.plot([i for i in range(2500)], loss[0])
    # plt.show()
    # fig1 = plt.figure(figsize=(20, 240))
    # plt.plot([i for i in range(2500)], new_add_signal[0])
    # plt.plot([i for i in range(2500)], depression_data1[0])
    # plt.show()
    # 结论，傅里叶变换结果复原与原函数差距很大，不建议使用傅里叶函数进行复原
