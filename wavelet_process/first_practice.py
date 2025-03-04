# import torch
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from tqdm import tqdm
# from Config import ConfigSet, set_seed
#
#
# class FeatureGetNet(nn.Module):
#     def __init__(self, config_):
#         super(FeatureGetNet, self).__init__()
#         self.config = config_
#         self.linear = nn.Linear(config_['relate_number'] - 1, 1)
#         # self.net = nn.Sequential()
#         # self.config = config_
#         # for i in range(len(config_['linear_dim']) - 1):
#         #   self.net.add_module('linear{}'.format(i), nn.Linear(config_['linear_dim'][i], config_['linear_dim'][i + 1]))
#
#     def forward(self, data_input):
#         data_input = self.linear(data_input.view(data_input.shape[0], data_input.shape[2], data_input.shape[1]))
#         data_input = data_input.view(data_input.shape[0], 1, data_input.shape[1])
#         output = 0
#         for i in range(data_input.shape[1]):
#             output += data_input[:, i, :]
#         return torch.tensor(output / data_input.shape[0] / 0.8).reshape(data_input.shape[0], 1, data_input.shape[2])
#         # shape = data_input.shape
#         # data_input = data_input.reshape(shape[0], shape[2], shape[1])
#         # data_input = self.net(data_input)
#         # return data_input.reshape(shape[0], self.config['linear_dim'][-1], self.config['hidden_dim'][0])
#
#
# class UNet(nn.Module):
#     def __init__(self, config_):
#         """
#         这里是一个基于全连接生成的U_net
#         :param config_:配置文件，其中必须包括hidden_dim
#         """
#         super(UNet, self).__init__()
#         self.net = nn.Sequential()
#         self.config = config_
#         if not len(config_['hidden_dim']) >= 2:
#             raise ValueError("config文件中必须保证hidden_dim大于等于2")
#         for j in range(config_['u_net number']):
#             for i in range(len(config_['hidden_dim']) - 1):
#                 self.net.add_module("linear{}".format(i), nn.Linear(config_['hidden_dim'][i],
#                                                                     config_['hidden_dim'][i + 1], device=config_['device']))
#                 if i != len(config_['hidden_dim']) - 1:
#                     self.net.add_module("gelu{}".format(i), nn.GELU())
#
#     def forward(self, input_data):
#         self.net.to(device=self.config['device'])
#         input_data.to(self.config['device'])
#         if not torch.is_tensor(input_data):
#             raise ValueError("请将输入数据转化为tensor类型")
#         if not input_data.shape[-1] == self.config['hidden_dim'][0]:
#             print(input_data.shape[-1], config['hidden_dim'][0])
#             raise ValueError("输入数据的维度与全连接维度不匹配")
#         output = self.net(input_data)
#         output = torch.max_pool2d(output, kernel_size=(output.shape[-2], 1), stride=(1, 1))
#         return output
#
#
# def train_function(data_packet_, config_):
#     device_ = config_['device']
#     dp_cA, dp_cD = data_packet_[0]
#     dp_cD *= 100
#     ct_cA, ct_cD = data_packet_[1]
#     ct_cD *= 100
#     dp_val_cA, dp_val_cD = data_packet_[2]
#     dp_val_cD *= 100
#     ct_val_cA, ct_val_cD = data_packet_[3]
#     ct_val_cD *= 100
#     depression_cnn = FeatureGetNet(config_).to(device_)
#     control_cnn = FeatureGetNet(config_).to(device_)
#     depression_u_net = UNet(config_).to(device_)
#     control_u_net = UNet(config_).to(device_)
#     depression_optimizer = torch.optim.Adam([{'params': depression_u_net.parameters()},
#                                              {'params': depression_cnn.parameters()}], lr=config_['lr'])
#     control_optimizer = torch.optim.Adam([{'params': control_u_net.parameters()},
#                                           {'params': control_cnn.parameters()}], lr=config_['lr'])
#     loss_f = nn.MSELoss()
#     dd_loss_list = []
#     dc_loss_list = []
#     cc_loss_list = []
#     cd_loss_list = []
#     for i in range(config_['epoch']):
#         with tqdm(total=dp_cA.shape[0]) as pbar:
#             pbar.set_description("Epoch{} Process".format(i))
#             for j in range(dp_cA.shape[0]):
#                 depression_optimizer.zero_grad()
#                 control_optimizer.zero_grad()
#                 dp_op = depression_cnn(dp_cA[j, :, 1:, :].to(device_))
#                 ct_op = control_cnn (ct_cA[j, :, 1:, :].to(device_))
#                 dp_op = depression_u_net(dp_op)
#                 ct_op = control_u_net(ct_op)
#                 dp_loss = loss_f(dp_op, dp_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 ct_loss = loss_f(ct_op, ct_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 dp_loss.sum().backward()
#                 ct_loss.sum().backward()
#                 depression_optimizer.step()
#                 control_optimizer.step()
#                 pbar.update(1)
#         with tqdm(total=dp_val_cA.shape[0] + ct_val_cA.shape[0]) as pbar:
#             pbar.set_description("Depression Val")
#             dd_add_loss = 0
#             dc_add_loss = 0
#             cc_add_loss = 0
#             cd_add_loss = 0
#             for j in range(dp_val_cA.shape[0]):
#                 d_dp_op = depression_cnn(dp_val_cA[j, :, 1:, :].to(device_))
#                 c_dp_op = control_cnn(dp_val_cA[j, :, 1:, :].to(device_))
#                 d_dp_op = depression_u_net(d_dp_op)
#                 c_dp_op = control_u_net(c_dp_op)
#                 d_dp_loss = loss_f(d_dp_op, dp_val_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 c_dp_loss = loss_f(c_dp_op, dp_val_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 dd_add_loss += d_dp_loss.sum().detach().item()
#                 cd_add_loss += c_dp_loss.sum().detach().item()
#                 pbar.update(1)
#             for j in range(ct_val_cA.shape[0]):
#                 d_ct_op = depression_cnn(ct_val_cA[j, :, 1:, :].to(device_))
#                 c_ct_op = control_cnn(ct_val_cA[j, :, 1:, :].to(device_))
#                 d_ct_op = depression_u_net(d_ct_op)
#                 c_ct_op = control_u_net(c_ct_op)
#                 d_ct_loss = loss_f(d_ct_op, ct_val_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 c_ct_loss = loss_f(c_ct_op, ct_val_cA[j, :, 0, :].to(device_)) / config_['batch_size']
#                 dc_add_loss += d_ct_loss.sum().detach().item()
#                 cc_add_loss += c_ct_loss.sum().detach().item()
#                 pbar.update(1)
#             dd_loss_list.append(dd_add_loss / dp_val_cA.shape[0])
#             dc_loss_list.append(dc_add_loss / ct_val_cA.shape[0])
#             cc_loss_list.append(cc_add_loss / ct_val_cA.shape[0])
#             cd_loss_list.append(cd_add_loss / dp_val_cA.shape[0])
#     l1, = plt.plot([i for i in range(config_['epoch'])], dd_loss_list, color='r')
#     l2, = plt.plot([i for i in range(config_['epoch'])], cd_loss_list, color='g')
#     plt.title("depression data")
#     plt.legend(handles=[l1, l2], labels=['depression segment net', 'control segment net'])
#     plt.show()
#     l3, = plt.plot([i for i in range(config_['epoch'])], cc_loss_list, color='g')
#     l4, = plt.plot([i for i in range(config_['epoch'])], dc_loss_list, color='r')
#     plt.title("control data")
#     plt.legend(handles=[l3, l4], labels=['control segment net', 'depression segment net'])
#     plt.show()
#
#
# if __name__ == '__main__':
#     set_seed(42)
#     data = torch.load(r"G:\python program\brain_relate_net\wavelet first try\middle_data\processed_data.pt")
#     config = ConfigSet()
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     config.add_item({"device": device,
#                      "hidden_dim": [1255, 1024, 512, 256, 512, 1024, 1255],
#                      "linear_dim": [5, 10, 5, 3, 1],
#                      "wt_family": 'db6',
#                      "lr": 0.00003,
#                      "u_net number": 4,
#                      "epoch": 20,
#                      "bias": [1, 0.9, 0.8, 0.7, 0.6],
#                      "time_len": 2500,
#                      "channel_number": 128,
#                      'relate_number': 6,
#                      'modma_data_root': r"G:\data package\MODMA_data\data_8trial.mat",
#                      'batch_size': 32})
#     train_function(data['processed_data'], config)
