import torch
import random
import os

from Config import ConfigSet, set_seed
from train import model_safer, middle_info_safer, depression_train, control_train, tst


if __name__ == '__main__':
    config = ConfigSet(conv_mean=0, conv_std=0.5, linear_mean=0, linear_std=0.5,
                       depression_train_data_path=r"G:\data package\MDD_data\wavelet_construct_data\depression_train_data.pt",
                       depression_val_data_path="G:\data package\MDD_data\wavelet_construct_data\depression_val_data.pt",
                       depression_test_data_path="G:\data package\MDD_data\wavelet_construct_data\depression_test_data.pt",
                       control_train_data_path="G:\data package\MDD_data\wavelet_construct_data\control_train_data.pt",
                       control_val_data_path="G:\data package\MDD_data\wavelet_construct_data\control_val_data.pt",
                       control_test_data_path="G:\data package\MDD_data\wavelet_construct_data\control_test_data.pt",
                       batch_size=19,
                       epoch_number=30, lr=0.00005,
                       family='db6',
                       input_channel=[1, 4, 10, 6, 4, 2],
                       hidden_channel=[4, 10, 6, 4, 2, 1],
                       kernel_size=[(2, 4), (2, 4), (2, 4), (1, 4), (1, 4), (1, 4)],
                       stride=[(2, 2), (2, 2), (2, 1), (1, 1), (1, 1), (1, 1)],
                       input_linear_dim=[307, 1024, 1285],
                       hidden_linear_dim=[1024, 1285, 1285],
                       middle_info_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information",
                       model_param_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters")

    depression_cA_net, depression_cD_net, depression_class_net = depression_train(config_=config)
    control_cA_net, control_cD_net, control_class_net = control_train(config_=config)
