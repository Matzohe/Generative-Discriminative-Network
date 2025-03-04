import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from Config import ConfigSet
from cnn_process import ConvolutionNet, ClassificationNet
from ftt_wavelet_dataloader import FftWaveletDataset
from wavelet_reconstruct import WaveletReconstruct
from Config import set_seed

set_seed(42)


def middle_info_safer(middle_infomation, info_name, config_):
    save_root = config_['new_middle_info_save_root']
    save_root = os.path.join(save_root, info_name + '.pt')
    # save_dict = {info_name: middle_infomation}
    # 这里修改了一下torch.save保存的内容，之前一直是用字典存的，现在发现原来可以直接存矩阵，亏麻了
    torch.save(middle_infomation, save_root)


def val_tst(config_, depression_cA_net, depression_cD_net, depression_class_net,
        control_cA_net, control_cD_net, control_class_net):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depression_test_dataset = FftWaveletDataset(config_['depression_val_data_path'], device)
    control_test_dataset = FftWaveletDataset(config_['control_val_data_path'], device)
    # 这三个list是用于保存original,depression和control分类器产生的cA和cD的
    original_depression_save_list = []
    depression_depression_save_list = []
    control_depression_save_list = []
    original_control_save_list = []
    depression_control_save_list = []
    control_control_save_list = []
    # 后续的曲线绘制中，depression的分类器是红色的，而control的分类器是绿色的
    with torch.no_grad():
        with tqdm(total=len(depression_test_dataset)) as pbar:
            pbar.set_description("depression data test")
            depression_list = []
            control_list = []
            depression_new_list = []
            control_new_list = []
            for i in range(len(depression_test_dataset)):
                cA, cD = depression_test_dataset[i]
                cA_test, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_test, cD_label = cD[:, 1:, :], cD[:, 0, :]
                # 保存一下原始信号
                original_depression_save_list.append((cA_label, cD_label))

                depression_cA_output = depression_cA_net(cA_test.unsqueeze(1))
                depression_cD_output = depression_cD_net(cD_test.unsqueeze(1))
                # 这里的信息保存到额外的list中，之后对这段信息进行保存,同时，注意这里修改了net部分的信息。
                depression_cA_result, depression_cD_result, depression_hidden = depression_class_net(
                    (depression_cA_output, depression_cD_output))
                depression_new_list.append(depression_hidden.detach())
                # 保存一下depression分类器生成的cA和cD
                depression_depression_save_list.append((depression_cA_result, depression_cD_result))

                control_cA_output = control_cA_net(cA_test.unsqueeze(1))
                control_cD_output = control_cD_net(cD_test.unsqueeze(1))
                control_cA_result, control_cD_result, control_hidden = control_class_net(
                    (control_cA_output, control_cD_output))
                control_new_list.append(control_hidden.detach())

                # 保存一下control分类器生成的cA和cD
                control_depression_save_list.append((control_cA_result, control_cD_result))

                depression_loss = WaveletReconstruct(config_['family'], depression_cA_result, depression_cD_result,
                                                     cA_label, cD_label)
                control_loss = WaveletReconstruct(config_['family'], control_cA_result, control_cD_result, cA_label,
                                                  cD_label)
                pbar.update(1)
                depression_list.append(depression_loss)
                control_list.append(control_loss)
            depression_list = torch.cat(depression_list, dim=0).view(-1)
            control_list = torch.cat(control_list, dim=0).view(-1)

            depression_new_list = torch.cat(depression_new_list, dim=0).view(-1, 128, 1, 300)
            control_new_list = torch.cat(control_new_list, dim=0).view(-1, 128, 1, 300)
            print(control_new_list.shape)
            new_list = torch.cat([depression_new_list, control_new_list], dim=-2)
            middle_info_safer(new_list, "depression/val", config_)
            plt.plot([j for j in range(len(depression_test_dataset) * config_['batch_size'])], depression_list,
                     color='r')
            plt.plot([j for j in range(len(depression_test_dataset) * config_['batch_size'])], control_list,
                     color='g')
            plt.show()
            plt.scatter([j for j in range(len(depression_test_dataset) * config_['batch_size'])],
                        [depression_list[i] - control_list[i]
                         for i in range(len(depression_test_dataset) * config_['batch_size'])], color='y', s=2)
            plt.show()
        with tqdm(total=len(control_test_dataset)) as pbar:
            pbar.set_description("control data test")
            depression_list = []
            control_list = []
            depression_new_list = []
            control_new_list = []
            for i in range(len(control_test_dataset)):
                cA, cD = control_test_dataset[i]
                cA_test, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_test, cD_label = cD[:, 1:, :], cD[:, 0, :]

                original_control_save_list.append((cA_label, cD_label))

                depression_cA_output = depression_cA_net(cA_test.unsqueeze(1))
                depression_cD_output = depression_cD_net(cD_test.unsqueeze(1))
                depression_cA_result, depression_cD_result, depression_c_hidden = depression_class_net(
                    (depression_cA_output, depression_cD_output))
                depression_new_list.append(depression_c_hidden)
                depression_control_save_list.append((depression_cA_result, depression_cD_result))

                control_cA_output = control_cA_net(cA_test.unsqueeze(1))
                control_cD_output = control_cD_net(cD_test.unsqueeze(1))
                control_cA_result, control_cD_result, control_c_hidden = control_class_net(
                    (control_cA_output, control_cD_output))
                control_new_list.append(control_c_hidden)

                control_control_save_list.append((control_cA_result, control_cD_result))

                depression_loss = WaveletReconstruct(config_['family'], depression_cA_result, depression_cD_result,
                                                     cA_label, cD_label)
                control_loss = WaveletReconstruct(config_['family'], control_cA_result, control_cD_result, cA_label,
                                                  cD_label)
                depression_list.append(depression_loss)
                control_list.append(control_loss)
                pbar.update(1)
            depression_list = torch.cat(depression_list, dim=0).view(-1)
            control_list = torch.cat(control_list, dim=0).view(-1)
            depression_new_list = torch.cat(depression_new_list, dim=0).view(-1, 128, 1, 300)
            control_new_list = torch.cat(control_new_list, dim=0).view(-1, 128, 1, 300)
            new_list = torch.cat([depression_new_list, control_new_list], dim=-2)
            print(new_list.shape)
            middle_info_safer(new_list, "control/val", config_)
            plt.plot([j for j in range(len(control_test_dataset) * config_['batch_size'])], depression_list,
                     color='r')
            plt.plot([j for j in range(len(control_test_dataset) * config_['batch_size'])], control_list,
                     color='g')
            plt.show()
            plt.scatter([j for j in range(len(control_test_dataset) * config_['batch_size'])],
                        [control_list[i] - depression_list[i]
                         for i in range(len(control_test_dataset) * config_['batch_size'])], color='y', s=2)
            plt.show()
    # middle_info_safer(original_depression_save_list, "val_original_depression_save_wavelet_list_200epoch", config_)
    # middle_info_safer(depression_depression_save_list, "val_depression_depression_save_wavelet_list_200epoch", config_)
    # middle_info_safer(control_depression_save_list, "val_control_depression_save_wavelet_list_200epoch", config_)
    # middle_info_safer(original_control_save_list, "val_original_control_wavelet_list_200epoch", config_)
    # middle_info_safer(depression_control_save_list, "val_depression_control_save_wavelet_list_200epoch", config_)
    # middle_info_safer(control_control_save_list, "val_control_control_save_wavelet_list_200epoch", config_)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ConfigSet(conv_mean=0, conv_std=0.5, linear_mean=0, linear_std=0.5,
                       depression_train_data_path=r"G:\python_program\middle_data\fft_wavelet_depression_train_data.pt",
                       depression_val_data_path=r"G:\python_program\middle_data\fft_wavelet_depression_val_data.pt",
                       depression_test_data_path=r"G:\python_program\middle_data\fft_wavelet_depression_test_data.pt",
                       control_train_data_path=r"G:\python_program\middle_data\fft_wavelet_control_train_data.pt",
                       control_val_data_path=r"G:\python_program\middle_data\fft_wavelet_control_val_data.pt",
                       control_test_data_path=r"G:\python_program\middle_data\fft_wavelet_control_test_data.pt",
                       batch_size=32,
                       epoch_number=200, lr=0.00005,
                       family='db6',
                       input_channel=[1, 4, 10, 6, 4, 2],
                       hidden_channel=[4, 10, 6, 4, 2, 1],
                       kernel_size=[(2, 4), (2, 4), (2, 4), (1, 4), (1, 4), (1, 4)],
                       stride=[(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
                       input_linear_dim=[300, 512, 1024, 1255],
                       hidden_linear_dim=[512, 1024, 1255, 1255],
                       middle_info_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\test_information",
                       model_param_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters",
                       new_middle_info_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\another_information",)
    depression_cA_net = ConvolutionNet(config)
    depression_cA_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\depression_cA_conv_net_200epoch_parameters.pt")
    depression_cA_net.load_state_dict(depression_cA_dict)
    depression_cD_net = ConvolutionNet(config)
    depression_cD_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\depression_cD_conv_net_200epoch_parameters.pt")
    depression_cD_net.load_state_dict(depression_cD_dict)
    depression_class_net = ClassificationNet(config)
    depression_class_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\depression_class_net_200epoch_parameters.pt")
    depression_class_net.load_state_dict(depression_class_dict)
    control_cA_net = ConvolutionNet(config)
    control_cA_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\control_cA_conv_net_200epoch_parameters.pt")
    control_cA_net.load_state_dict(control_cA_dict)
    control_cD_net = ConvolutionNet(config)
    control_cD_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\control_cD_conv_net_200epoch_parameters.pt")
    control_cD_net.load_state_dict(control_cD_dict)
    control_class_net = ClassificationNet(config)
    control_class_dict = torch.load(r"G:\python_program\neuro_network_information\brain_relate_net\model_parameters\control_class_net_200epoch_parameters.pt")
    control_class_net.load_state_dict(control_class_dict)
    depression_cA_net.to(device)
    depression_cD_net.to(device)
    depression_class_net.to(device)
    control_cA_net.to(device)
    control_cD_net.to(device)
    control_class_net.to(device)
    val_tst(config, depression_cA_net, depression_cD_net, depression_class_net,
            control_cA_net, control_cD_net, control_class_net)
