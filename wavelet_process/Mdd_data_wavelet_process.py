import torch
import random
import os
from tqdm import tqdm

from modma_data_wavelet_process import MODMA_process as MDD_process
from Config import ConfigSet, set_seed
from MddDataloader import MddDataLoader
set_seed(42)


if __name__ == '__main__':

    train_list = [i for i in range(15)]
    random.shuffle(train_list)
    test_list = [i for i in range(15, 20)]
    random.shuffle(test_list)
    depression_val_list = [i for i in range(20, 29)]
    random.shuffle(depression_val_list)
    control_val_list = [i for i in range(20, 25)]
    random.shuffle(control_val_list)
    map_list = [i for i in range(14)]
    random.shuffle(map_list)
    config = ConfigSet()
    config.add_item({"device": 'cpu',
                     "wt_family": 'db6',
                     "time_len": 2560,
                     "channel_number": 19,
                     'train_list': train_list,
                     'test_list': test_list,
                     'depression_val': depression_val_list,
                     'control_val': control_val_list,
                     'map_list': map_list,
                     'relate_number': 11,
                     'mdd_data_root': r"G:\data package\MDD_data\my_mdd",
                     'batch_size': 19,
                     'STFT_information': {'time_len': 2560,
                                          'fstart': [1, 4, 8, 14, 31],
                                          'fend': [3, 7, 13, 30, 50],
                                          'foriginal': 256,
                                          'window': 10}})
    depression_dataset = MddDataLoader(config['mdd_data_root'], 'depression', device=config['device'])
    control_dataset = MddDataLoader(config['mdd_data_root'], 'control', device=config['device'])
    depression_train_data, control_train_data, depression_val_data, control_val_data, depression_test_data,\
        control_test_data = MDD_process(depression_dataset, control_dataset, config)
    save_path = r"G:\data package\MDD_data\wavelet_construct_data"
    torch.save(depression_train_data, os.path.join(save_path, 'depression_train_data.pt'))
    torch.save(control_train_data, os.path.join(save_path, 'control_train_data.pt'))
    torch.save(depression_val_data, os.path.join(save_path, 'depression_val_data.pt'))
    torch.save(control_val_data, os.path.join(save_path, 'control_val_data.pt'))
    torch.save(depression_test_data, os.path.join(save_path, 'depression_test_data.pt'))
    torch.save(control_test_data, os.path.join(save_path, 'control_test_data.pt'))
