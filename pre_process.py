import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from collections import OrderedDict
import torch.optim as optim
import os
from tqdm import tqdm
import mne
import math
import random


np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class my_dataset(Dataset):
    def __init__(self, data_root, label):
        """
        mdd channel name and order information
        这里1-8和10-17是左右脑对称电极
        'EEG Fp1-LE': 'Fp1', 'EEG F3-LE': 'F3', 'EEG C3-LE': 'C3', 'EEG P3-LE': 'P3', 'EEG O1-LE': 'O1',
        'EEG F7-LE': 'F7', 'EEG T3-LE': 'T3', 'EEG T5-LE': 'T5', 'EEG Fz-LE': 'Fz', 'EEG Fp2-LE': 'Fp2',
        'EEG F4-LE': 'F4', 'EEG C4-LE': 'C4', 'EEG P4-LE': 'P4', 'EEG O2-LE': 'O2', 'EEG F8-LE': 'F8',
        'EEG T4-LE': 'T4', 'EEG T6-LE': 'T6', 'EEG Cz-LE': 'Cz', 'EEG Pz-LE': 'Pz', 'EEG A2-A1': 'A1'
        """
        super(my_dataset, self).__init__()
        self.label = label
        self.datadir = os.listdir(os.path.join(data_root, label))
        self.data_root = os.path.join(data_root, label)

    def __getitem__(self, index):
        try:
            data_path = os.path.join(self.data_root, self.datadir[index])
            output_data = torch.load(data_path)
        except IOError:
            raise RuntimeError('IOError')
        return output_data['data'] * 100000


def attention_node(q, k):
    if q.ndim < 3 and k.ndim < 3:
        return q @ k.T
    else:
        raise RuntimeError("attention_node暂时不支持三维的输入")


def relation_culculate_function(env_array, key_array):
    """
    这个函数是用来计算距离的，第一个参数是被对比的对象，第二个参数是对比的对象
    """
    # 计算余弦距离，输入的数据是tensor，先转化为numpy
    # 现在这个函数可以处理二维的数据输入
    # 中间的处理可以利用 tensor 的unsqueeze把env_array从中间扩大一维，利用扩散机制去完成中间的加和计算
    env_array = env_array.numpy()
    key_array = key_array.numpy()
    env_shape = env_array.shape
    env_dim = env_array.ndim
    key_shape = key_array.size
    key_dim = key_array.ndim
    # 这里对输入数据的形状做一次检查,对不同形状的输入有不同的处理方式
    if (env_dim == 1 or env_dim == 2) and (key_dim == 2 or key_dim == 1):
        if env_dim == 1:
            env_array.reshape(1, env_array.shape[0])
        if key_dim == 1:
            key_array.reshape(1, key_array.shape[0])
        if env_dim == 1:
            cos_similarity = np.sum((env_array * key_array), axis=-1) / np.linalg.norm(env_array, ord=2,
                                                                                       axis=-1) / np.linalg.norm(
                key_array, ord=2, axis=1)
        else:
            cos_similarity = []
            for each in range(env_shape[0]):
                cos_similarity.append(
                    np.sum((env_array[each] * key_array), axis=-1) / np.linalg.norm(env_array[each], ord=2,
                                                                                    axis=-1) / np.linalg.norm(key_array,
                                                                                                              ord=2,
                                                                                                              axis=-1))
        # 计算曼哈顿距离,利用python的广播机制
        if env_dim == 2:
            manhattan_distance = []
            for each in range(env_shape[0]):
                manhattan_distance.append(np.sum(np.abs(env_array[each] - key_array), axis=1))
        else:
            manhattan_distance = np.sum(np.abs(env_array - key_array), axis=1)
        # 通过计算attention来看数据之间的相似度
        attention_distance = attention_node(env_array, key_array) / math.sqrt(env_shape[-1])
        # 计算二阶范数
        if env_dim == 1:
            archimedes_distance = np.linalg.norm(env_array - key_array)
        else:
            archimedes_distance = []
            for each in range(env_shape[0]):
                archimedes_distance.append(np.linalg.norm(env_array[each] - key_array, axis=1))
        return torch.tensor(np.array(cos_similarity)), torch.tensor(np.array(manhattan_distance)), torch.tensor(
            np.array(attention_distance)), torch.tensor(np.array(archimedes_distance))
    else:
        raise RuntimeError("在计算相似度时(relation_culculate_function)请输入二维的数据，三维数据的处理并没有写")

# 接下去根据不同的数据去预测脑电，比如，我想要复原一个电极，取这个电极对面的和相似度高的4个电极，训练的时候根据相似度排列，测试的问题再说。用256个点（采样频率）作为一个batch

def mdd_channel_select(input_data):
    """
    将输入的脑电数据分为左脑、右脑与中间脑
    输入数据的电极安放方法是根据mdd数据集的信息确定的，输入总的channel数为19个
    """
    left_head_channel = input_data[: 8]
    right_head_channel = input_data[9: 17]
    central_channel = torch.cat([input_data[8], input_data[17], input_data[18]], dim=0)
    return left_head_channel, right_head_channel, central_channel


def select_channel_for_predict(select_information, index, max_number):
    index_list = []  # 返回排序筛选之后的index
    similarity_list = []
    # 这里写得很臭，中间那个循环可以优化掉
    # 最后返回的是max_number个index
    i = 0
    for each in range(index):
        final_list = torch.cat([select_information[i, :].view(-1)])
        _, output_list = torch.sort(final_list, descending=True)
        new_list = [k for k in output_list[: max_number]]
        # if each < 8:
        #     new_list.append(each + 9)
        # elif each >= 9 and each <= 16:
        #     new_list.append(each - 9)
        # else:
        #     new_list.append(each)
        index_list.append(torch.tensor(new_list))
        # similarity_list.append(torch.tensor(np.array(select_information[i, new_list])).view(-1))
        similarity_list.append(select_information[i, new_list])
        i += 1
    return index_list, similarity_list


def devide_eeg_data(input_data, batch_size, all_time_length):
    """
    将输入的数据切分为合理的小数据集，batch-size和总长度为超参数，可以自行设
    并且返回（batch_size, channel_number, time_sequence）的数据(二维情况下)
    如果输入为三维，则返回（第0维, batch_size, channel_number, time_sequence)
    """
    input_size = input_data.shape
    if len(input_size) == 2:
        """
        这里对于总长度与batch size不匹配的情况还没有写
        """
        input_data.view(-1, input_size[-1])
        input_data = input_data[:, : all_time_length]
        input_data = input_data.view(input_size[0], batch_size, -1)
        return input_data.permute(1, 0, 2)
    elif len(input_size) == 3:
        input_data.view(-1, input_size[-1])
        input_data = input_data[:, : all_time_length]
        input_data = input_data.view(input_size[0], input_size[1], batch_size, -1)
        return input_data.permute(0, 2, 1, 3)


def data_construct(depression_dataset, control_dataset, depression_data_number, control_data_number,
                   max_number, batch_size, all_time_length, channel_number):
    """
    将输入的数据根据余弦距离选择最接近的几组（max_number），重组数据
    depression_data_number 为被试的人数
    同样这个函数仅仅能够处理二维的数据，也就是一个被试的
    输入的数据为（channel_number, time_sequence）
    输出的数据为（总被试数，每个被试被分成的batch数，每个被试的总电极数，每个电极对应的最大关联电极数目，最大关联电极对应的脑电数据）
    """
    # 现将输入的数据分成batch_size的份数
    depression_data = []
    control_data = []
    for each in range(depression_data_number):
        input_data = depression_dataset[each]
        middle_data = devide_eeg_data(input_data[: channel_number], batch_size, all_time_length)
        for i in range(batch_size):
            contrast_information, _, _, _ = relation_culculate_function(middle_data[i], middle_data[i])
            input_index = [j for j in range(channel_number)]
            select_index_list, relation_list = select_channel_for_predict(contrast_information, input_index, max_number)
            per_batch_data = []
            for j in input_index:
                per_batch_data.append(middle_data[i][select_index_list[j]] * relation_list[j].unsqueeze(-1))
            per_batch_data = torch.cat(per_batch_data, dim=0)
            depression_data.append(per_batch_data)
    for each in range(control_data_number):
        input_data = control_dataset[each]
        middle_data = devide_eeg_data(input_data, batch_size, all_time_length)
        for i in range(batch_size):
            contrast_information, _, _, _ = relation_culculate_function(middle_data[i], middle_data[i])
            input_index = [j for j in range(channel_number)]
            select_index_list, relation_list = select_channel_for_predict(contrast_information, input_index, max_number)
            per_batch_data = []
            for j in input_index:
                per_batch_data.append(middle_data[i][select_index_list[j]] * relation_list[j].unsqueeze(-1))
            per_batch_data = torch.cat(per_batch_data, dim=0)
            control_data.append(per_batch_data)
    depression_data = torch.cat(depression_data, dim=0).view(depression_data_number, batch_size, channel_number,
                                                             max_number + 1, -1)
    control_data = torch.cat(control_data, dim=0).view(control_data_number, batch_size, channel_number, max_number + 1,
                                                       -1)
    return depression_data, control_data