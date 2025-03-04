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


def model_safer(model, model_name, config_):
    save_root = config_['model_param_save_root']
    model_name = model_name + "_parameters"
    save_root = os.path.join(save_root, model_name + '.pt')
    torch.save(model.state_dict(), save_root)

def middle_info_safer(middle_infomation, info_name, config_):
    save_root = config_['middle_info_save_root']
    save_root = os.path.join(save_root, info_name + '.pt')
    torch.save(middle_infomation, save_root)


def init_process(model, config_):
    # 参数初始化过程，其中conv和linear的weight参数可以在config文件中修改，而对这些层的bias默认初始为0
    for each in model.modules():
        if isinstance(each, nn.Conv2d):
            nn.init.normal_(each.weight, mean=config_['conv_mean'], std=config_['conv_std'])
            nn.init.zeros_(each.bias)
        elif isinstance(each, nn.Linear):
            nn.init.normal_(each.weight, mean=config_['linear_mean'], std=config_['linear_std'])
            nn.init.zeros_(each.bias)


def depression_train(config_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depression_train_dataset = FftWaveletDataset(config_['depression_train_data_path'], device)
    # depression_val_dataset = FftWaveletDataset(config_['depression_val_data_path'], device)
    # control_val_dataset = FftWaveletDataset(config_['control_val_data_path'], device)
    epoch_number = config_['epoch_number']

    cA_conv_net = ConvolutionNet(config_)
    init_process(cA_conv_net, config_)

    cD_conv_net = ConvolutionNet(config_)
    init_process(cD_conv_net, config_)

    class_net = ClassificationNet(config_)
    init_process(class_net, config_)

    cA_conv_net.to(device)
    cD_conv_net.to(device)
    class_net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam([{'params': cA_conv_net.parameters()},
                            {'params': cD_conv_net.parameters()},
                            {'params': class_net.parameters()}], lr=config_['lr'])
    loss_info = []
    for each in range(epoch_number):
        with tqdm(total=len(depression_train_dataset)) as qbar:
            qbar.set_description("training epoch" + str(each + 1))
            all_loss = 0
            for i in range(len(depression_train_dataset)):
                optimizer.zero_grad()
                cA, cD = depression_train_dataset[i]
                cA_train, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_train, cD_label = cD[:, 1:, :], cD[:, 0, :]
                cA_output = cA_conv_net(cA_train.unsqueeze(1))
                cD_output = cD_conv_net(cD_train.unsqueeze(1))
                cA_result, cD_result = class_net((cA_output, cD_output))
                loss = (loss_function(cA_result, cA_label) + loss_function(cD_result, cD_label)) / torch.tensor(
                    cA.shape[0], device=device)
                loss.requires_grad_(True)
                loss.sum().backward()
                all_loss += loss.detach().to('cpu')
                optimizer.step()
                qbar.update(1)
            loss_info.append(all_loss / len(depression_train_dataset) / config_['batch_size'])

        # if (each + 1) % 10 == 0:
        #     d_loss_list = []
        #     c_loss_list = []
        #     for i in range(len(depression_val_dataset)):
        #         with torch.no_grad():
        #             depression_cA, depression_cD = depression_val_dataset[i]
        #             d_cA_train, d_cA_label = depression_cA[:, 1:, :], depression_cA[:, 0, :]
        #             d_cD_train, d_cD_label = depression_cD[:, 1:, :], depression_cD[:, 0, :]
        #             d_cA_output = cA_conv_net(d_cA_train.unsqueeze(1))
        #             d_cD_output = cD_conv_net(d_cD_train.unsqueeze(1))
        #             d_cA_result, d_cD_result = class_net((d_cA_output, d_cD_output))
        #             # 这里之前是使用MSEloss进行损失估计，但是认为，应该是需要进行reconstruct之后再计算损失才是合理的
        #             # d_loss = loss_function(d_cA_result, d_cA_label) + \
        #             #          loss_function(d_cD_result, d_cD_label) / cA.shape[0]
        #             # d_loss_list.append(d_loss.sum().detach().to('cpu'))
        #             d_loss = WaveletReconstruct(config_['family'], d_cA_result, d_cD_result, d_cA_label, d_cD_label)
        #             d_loss_list.append(d_loss)
        #             control_cA, control_cD = control_val_dataset[i]
        #             c_cA_train, c_cA_label = control_cA[:, 1:, :], control_cA[:, 0, :]
        #             c_cD_train, c_cD_label = control_cD[:, 1:, :], control_cD[:, 0, :]
        #             c_cA_output = cA_conv_net(c_cA_train.unsqueeze(1))
        #             c_cD_output = cD_conv_net(c_cD_train.unsqueeze(1))
        #             c_cA_result, c_cD_result = class_net((c_cA_output, c_cD_output))
        #             # c_loss = loss_function(c_cA_result, c_cA_label) + \
        #             #          loss_function(c_cD_result, c_cD_label) / cA.shape[0]
        #             # c_loss_list.append(c_loss.sum().detach().to('cpu'))
        #             c_loss = WaveletReconstruct(config_['family'], c_cA_result, c_cD_result, c_cA_label, c_cD_label)
        #             c_loss_list.append(c_loss)
        #     c_loss_list = torch.cat(c_loss_list, dim=0).view(-1)
        #     d_loss_list = torch.cat(d_loss_list, dim=0).view(-1)
        #     plt.scatter([j for j in range(len(depression_val_dataset) * config_['batch_size'])], d_loss_list, color='r', s=2)
        #     plt.scatter([j for j in range(len(depression_val_dataset) * config_['batch_size'])], c_loss_list, color='g', s=2)
        #     plt.show()
    model_safer(cA_conv_net, "depression_cA_conv_net_200epoch", config_)
    model_safer(cD_conv_net, "depression_cD_conv_net_200epoch", config_)
    model_safer(class_net, "depression_class_net_200epoch", config_)
    plt.plot([i for i in range(10, epoch_number)], loss_info[10:])
    plt.show()
    return cA_conv_net, cD_conv_net, class_net


def control_train(config_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    control_train_dataset = FftWaveletDataset(config_['control_train_data_path'], device)
    # control_val_dataset = FftWaveletDataset(config_['control_val_data_path'], device)
    # depression_val_dataset = FftWaveletDataset(config_['depression_val_data_path'], device)
    epoch_number = config_['epoch_number']

    cA_conv_net = ConvolutionNet(config_)
    init_process(cA_conv_net, config_)

    cD_conv_net = ConvolutionNet(config_)
    init_process(cD_conv_net, config_)

    class_net = ClassificationNet(config_)
    init_process(class_net, config_)

    cA_conv_net.to(device)
    cD_conv_net.to(device)
    class_net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam([{'params': cA_conv_net.parameters()},
                            {'params': cD_conv_net.parameters()},
                            {'params': class_net.parameters()}], lr=config_['lr'])
    loss_info = []
    for each in range(epoch_number):
        with tqdm(total=len(control_train_dataset)) as qbar:
            qbar.set_description("training epoch" + str(each + 1))
            all_loss = 0
            for i in range(len(control_train_dataset)):
                optimizer.zero_grad()
                cA, cD = control_train_dataset[i]
                cA_train, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_train, cD_label = cD[:, 1:, :], cD[:, 0, :]
                cA_output = cA_conv_net(cA_train.unsqueeze(1))
                cD_output = cD_conv_net(cD_train.unsqueeze(1))
                cA_result, cD_result = class_net((cA_output, cD_output))
                loss = (loss_function(cA_result, cA_label) + loss_function(cD_result, cD_label)) / torch.tensor(
                    cA.shape[0], device=device)
                loss.requires_grad_(True)
                loss.sum().backward()
                all_loss += loss.detach().to('cpu')
                optimizer.step()
                qbar.update(1)
            loss_info.append(all_loss / len(control_train_dataset) / config_['batch_size'])
    plt.plot([i for i in range(10, epoch_number)], loss_info[10:])
    plt.show()
    model_safer(cA_conv_net, "control_cA_conv_net_200epoch", config_)
    model_safer(cD_conv_net, "control_cD_conv_net_200epoch", config_)
    model_safer(class_net, "control_class_net_200epoch", config_)
    return cA_conv_net, cD_conv_net, class_net


def tst(config_, depression_cA_net, depression_cD_net, depression_class_net,
        control_cA_net, control_cD_net, control_class_net):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depression_test_dataset = FftWaveletDataset(config_['depression_test_data_path'], device)
    control_test_dataset = FftWaveletDataset(config_['control_test_data_path'], device)
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
            for i in range(len(depression_test_dataset)):
                cA, cD = depression_test_dataset[i]
                cA_test, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_test, cD_label = cD[:, 1:, :], cD[:, 0, :]
                # 保存一下原始信号
                original_depression_save_list.append((cA_label, cD_label))

                depression_cA_output = depression_cA_net(cA_test.unsqueeze(1))
                depression_cD_output = depression_cD_net(cD_test.unsqueeze(1))
                depression_cA_result, depression_cD_result = depression_class_net(
                    (depression_cA_output, depression_cD_output))
                # 保存一下depression分类器生成的cA和cD
                depression_depression_save_list.append((depression_cA_result, depression_cD_result))

                control_cA_output = control_cA_net(cA_test.unsqueeze(1))
                control_cD_output = control_cD_net(cD_test.unsqueeze(1))
                control_cA_result, control_cD_result = control_class_net(
                    (control_cA_output, control_cD_output))
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
            for i in range(len(control_test_dataset)):
                cA, cD = control_test_dataset[i]
                cA_test, cA_label = cA[:, 1:, :], cA[:, 0, :]
                cD_test, cD_label = cD[:, 1:, :], cD[:, 0, :]

                original_control_save_list.append((cA_label, cD_label))

                depression_cA_output = depression_cA_net(cA_test.unsqueeze(1))
                depression_cD_output = depression_cD_net(cD_test.unsqueeze(1))
                depression_cA_result, depression_cD_result = depression_class_net(
                    (depression_cA_output, depression_cD_output))

                depression_control_save_list.append((depression_cA_result, depression_cD_result))

                control_cA_output = control_cA_net(cA_test.unsqueeze(1))
                control_cD_output = control_cD_net(cD_test.unsqueeze(1))
                control_cA_result, control_cD_result = control_class_net(
                    (control_cA_output, control_cD_output))

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
            plt.plot([j for j in range(len(control_test_dataset) * config_['batch_size'])], depression_list,
                     color='r')
            plt.plot([j for j in range(len(control_test_dataset) * config_['batch_size'])], control_list,
                     color='g')
            plt.show()
            plt.scatter([j for j in range(len(depression_test_dataset) * config_['batch_size'])],
                        [control_list[i] - depression_list[i]
                         for i in range(len(depression_test_dataset) * config_['batch_size'])], color='y', s=2)
            plt.show()
    middle_info_safer(original_depression_save_list, "original_save_wavelet_list_200epoch", config_)
    middle_info_safer(depression_depression_save_list, "depression_save_wavelet_list_200epoch", config_)
    middle_info_safer(control_depression_save_list, "control_save_wavelet_list_200epoch", config_)
    middle_info_safer(original_control_save_list, "original_control_wavelet_list_200epoch", config_)
    middle_info_safer(depression_control_save_list, "depression_control_save_wavelet_list_200epoch", config_)
    middle_info_safer(control_control_save_list, "control_control_save_wavelet_list_200epoch", config_)


if __name__ == '__main__':
    config = ConfigSet(conv_mean=0, conv_std=0.5, linear_mean=0, linear_std=0.5,
                       depression_train_data_path=r"G:\data package\MODMA_data\5_channel_selection\depression_train_data.pt",
                       depression_val_data_path=r"G:\data package\MODMA_data\5_channel_selection\depression_val_data.pt",
                       depression_test_data_path=r"G:\data package\MODMA_data\5_channel_selection\depression_test_data.pt",
                       control_train_data_path=r"G:\data package\MODMA_data\5_channel_selection\control_train_data.pt",
                       control_val_data_path=r"G:\data package\MODMA_data\5_channel_selection\control_val_data.pt",
                       control_test_data_path=r"G:\data package\MODMA_data\5_channel_selection\control_test_data.pt",
                       batch_size=32,
                       epoch_number=150, lr=0.00005,
                       family='db6',
                       input_channel=[1, 4, 10, 6, 4, 2],
                       hidden_channel=[4, 10, 6, 4, 2, 1],
                       kernel_size=[(2, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4)],
                       stride=[(2, 2), (1, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
                       input_linear_dim=[300, 512, 1024, 1255],
                       hidden_linear_dim=[512, 1024, 1255, 1255],
                       middle_info_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\middle_info",
                       model_param_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\MODMA_contrast_experiment\5_channel\model_info")

    depression_cA_net, depression_cD_net, depression_class_net = depression_train(config_=config)
    control_cA_net, control_cD_net, control_class_net = control_train(config_=config)
    tst(config, depression_cA_net, depression_cD_net, depression_class_net,
        control_cA_net, control_cD_net, control_class_net)
