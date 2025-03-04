import torch
from wavelet_process import wt_process
from Config import ConfigSet, set_seed
from tqdm import tqdm
from ModmaDataloader import MODMADataLoader
import random
import os
from pre_process import relation_culculate_function, select_channel_for_predict
from wavelet_reconstruct import WaveletReconstruct


def get_batch(data_input, config_):
    batch_size = config_['batch_size']
    data_input = torch.cat(data_input)
    cA_list = []
    cD_list = []
    with tqdm(total=data_input.shape[0]) as pbar:
        pbar.set_description("wavelet process")
        for i in range(data_input.shape[0]):
            cA1 = []
            cD1 = []
            cA, cD = wt_process(data_input[i], config_["wt_family"], config_)

            cA_list.append(cA)
            cD_list.append(cD)
            pbar.update(1)
    data_shape = cA_list[0].shape[-1]
    return torch.cat(cA_list, dim=-1).reshape(-1, batch_size, config_['relate_number'], data_shape),\
        torch.cat(cD_list, dim=-1).reshape(-1, batch_size, config_['relate_number'], data_shape)


def MODMA_process(depression_dataset_, control_dataset_, config_):
    depression_train_data = []
    control_train_data = []
    depression_val_data = []
    control_val_data = []
    depression_test_data = []
    control_test_data = []
    with tqdm(total=len(config_['train_list']) * len(config_['map_list'])) as pbar:
        pbar.set_description("Train Data Process")
        for i in config_['train_list']:
            for j in config_['map_list']:
                dp_dt = depression_dataset_[(i, j)]
                ct_dt = control_dataset_[(i, j)]
                dp_info, _, _, _ = relation_culculate_function(dp_dt, dp_dt)
                ct_info, _, _, _ = relation_culculate_function(ct_dt, dp_dt)
                dp_cn, _ = select_channel_for_predict(dp_info, config_['channel_number'], config_['relate_number'])
                ct_cn, _ = select_channel_for_predict(ct_info, config_['channel_number'], config_['relate_number'])
                dp_cn = torch.cat(dp_cn, dim=0).view(config_['channel_number'], -1).contiguous()
                ct_cn = torch.cat(ct_cn, dim=0).view(config_['channel_number'], -1).contiguous()
                dt_select = dp_dt[dp_cn]
                ct_select = ct_dt[ct_cn]
                depression_train_data.append(dt_select)
                control_train_data.append(ct_select)
                pbar.update(1)
    with tqdm(total=len(config_['test_list']) * len(config_['map_list'])) as pbar:
        pbar.set_description("Test Data Process")
        for i in config_['test_list']:
            for j in config_['map_list']:
                dp_dt = depression_dataset_[(i, j)]
                ct_dt = control_dataset_[(i, j)]
                dp_info, _, _, _ = relation_culculate_function(dp_dt, dp_dt)
                ct_info, _, _, _ = relation_culculate_function(ct_dt, dp_dt)
                dp_cn, _ = select_channel_for_predict(dp_info, config_['channel_number'], config_['relate_number'])
                ct_cn, _ = select_channel_for_predict(ct_info, config_['channel_number'], config_['relate_number'])
                dp_cn = torch.cat(dp_cn, dim=0).view(config_['channel_number'], -1).contiguous()
                ct_cn = torch.cat(ct_cn, dim=0).view(config_['channel_number'], -1).contiguous()
                dt_select = dp_dt[dp_cn]
                ct_select = ct_dt[ct_cn]
                depression_test_data.append(dt_select)
                control_test_data.append(ct_select)
                pbar.update(1)
    with tqdm(total=len(config_['depression_val']) * len(config_['map_list'])) as pbar:
        pbar.set_description("Depression Val Data Process")
        for i in config_['depression_val']:
            for j in config_['map_list']:
                dp_dt = depression_dataset_[(i, j)]
                dp_info, _, _, _ = relation_culculate_function(dp_dt, dp_dt)
                dp_cn, _ = select_channel_for_predict(dp_info, config_['channel_number'], config_['relate_number'])
                dp_cn = torch.cat(dp_cn, dim=0).view(config_['channel_number'], -1)
                dt_select = dp_dt[dp_cn]
                depression_val_data.append(dt_select)
                pbar.update(1)
    with tqdm(total=len(config_['control_val']) * len(config_['map_list'])) as pbar:
        pbar.set_description("Control Val Data Process")
        for i in config_['control_val']:
            for j in config_['map_list']:
                ct_dt = control_dataset_[(i, j)]
                ct_info, _, _, _ = relation_culculate_function(ct_dt, dp_dt)
                ct_cn, _ = select_channel_for_predict(ct_info, config_['channel_number'], config_['relate_number'])
                ct_cn = torch.cat(ct_cn, dim=0).view(config_['channel_number'], -1)
                ct_select = ct_dt[ct_cn]
                control_val_data.append(ct_select)
                pbar.update(1)
    return get_batch(depression_train_data, config_), get_batch(control_train_data, config_), \
        get_batch(depression_val_data, config_), get_batch(control_val_data, config_), \
        get_batch(depression_test_data, config_), get_batch(control_test_data, config_)


if __name__ == '__main__':

    """
    虽然数据处理过程中要求在cpu处理，但这里还是把gpu带上了
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    set_seed(42)
    train_list = [i for i in range(15)]
    random.shuffle(train_list)
    test_list = [i for i in range(15, 20)]
    random.shuffle(test_list)
    depression_val_list = [i for i in range(20, 24)]
    random.shuffle(depression_val_list)
    control_val_list = [i for i in range(20, 29)]
    random.shuffle(control_val_list)
    map_list = [i for i in range(8)]
    random.shuffle(map_list)
    config = ConfigSet()
    config.add_item({"device": device,
                     "wt_family": 'db6',
                     "time_len": 2500,
                     "channel_number": 128,
                     'train_list': train_list,
                     'test_list': test_list,
                     'depression_val': depression_val_list,
                     'control_val': control_val_list,
                     'map_list': map_list,
                     'relate_number': 6,
                     'modma_data_root': r"G:\data package\MODMA_data\data_8trial.mat",
                     'batch_size': 32,
                     'STFT_information': {'time_len': 2500,
                                          'fstart': [1, 4, 8, 14, 31],
                                          'fend': [3, 7, 13, 30, 50],
                                          'foriginal': 250,
                                          'window': 10}})
    depression_dataset = MODMADataLoader(config['modma_data_root'], 'depression')
    control_dataset = MODMADataLoader(config['modma_data_root'], 'control')
    # data_packet = MODMA_process(depression_dataset, control_dataset, config)  # data pack是一个元组
    # torch.save({'processed_data': data_packet},
    #            r"G:\python_program\brain_relate_net\wavelet first try\middle_data\fft_wavelet_processed_data.pt")
    depression_train_data, control_train_data, depression_val_data, control_val_data, depression_test_data, \
        control_test_data = MODMA_process(depression_dataset, control_dataset, config)
    save_path = r"G:\data package\MODMA_data\5_channel_selection"
    torch.save(depression_train_data, os.path.join(save_path, 'depression_train_data.pt'))
    torch.save(control_train_data, os.path.join(save_path, 'control_train_data.pt'))
    torch.save(depression_val_data, os.path.join(save_path, 'depression_val_data.pt'))
    torch.save(control_val_data, os.path.join(save_path, 'control_val_data.pt'))
    torch.save(depression_test_data, os.path.join(save_path, 'depression_test_data.pt'))
    torch.save(control_test_data, os.path.join(save_path, 'control_test_data.pt'))
