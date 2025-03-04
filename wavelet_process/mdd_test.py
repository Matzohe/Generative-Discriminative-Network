
import torch
from Config import ConfigSet, set_seed

from cnn_process import ConvolutionNet, ClassificationNet
from Modma_data_test import middle_info_safer, val_tst
set_seed(42)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ConfigSet(conv_mean=0, conv_std=0.5, linear_mean=0, linear_std=0.5,
                       depression_train_data_path=r"G:\data package\MDD_data\wavelet_construct_data\depression_train_data.pt",
                       depression_val_data_path="G:\data package\MDD_data\wavelet_construct_data\depression_val_data.pt",
                       depression_test_data_path="G:\data package\MDD_data\wavelet_construct_data\depression_test_data.pt",
                       control_train_data_path="G:\data package\MDD_data\wavelet_construct_data\control_train_data.pt",
                       control_val_data_path="G:\data package\MDD_data\wavelet_construct_data\control_val_data.pt",
                       control_test_data_path="G:\data package\MDD_data\wavelet_construct_data\control_test_data.pt",
                       batch_size=19,
                       epoch_number=20, lr=0.00005,
                       family='db6',
                       input_channel=[1, 4, 10, 6, 4, 2],
                       hidden_channel=[4, 10, 6, 4, 2, 1],
                       kernel_size=[(2, 4), (2, 4), (2, 4), (1, 4), (1, 4), (1, 4)],
                       stride=[(2, 2), (2, 2), (2, 1), (1, 1), (1, 1), (1, 1)],
                       input_linear_dim=[307, 1024, 1285],
                       hidden_linear_dim=[1024, 1285, 1285],
                       middle_info_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\middle_information",
                       model_param_save_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters",
                       depression_cA_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\depression_cA_conv_net_200epoch_parameters.pt",
                       depression_cD_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\depression_cD_conv_net_200epoch_parameters.pt",
                       depression_class_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\depression_class_net_200epoch_parameters.pt",
                       control_cA_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\control_cA_conv_net_200epoch_parameters.pt",
                       control_cD_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\control_cD_conv_net_200epoch_parameters.pt",
                       control_class_parameter_root=r"G:\python_program\neuro_network_information\brain_relate_net\mdd_data\model_parameters\control_class_net_200epoch_parameters.pt")
    depression_cA_net = ConvolutionNet(config)
    depression_cA_dict = torch.load(config['depression_cA_parameter_root'])
    depression_cA_net.load_state_dict(depression_cA_dict)
    depression_cD_net = ConvolutionNet(config)
    depression_cD_dict = torch.load(config['depression_cD_parameter_root'])
    depression_cD_net.load_state_dict(depression_cD_dict)
    depression_class_net = ClassificationNet(config)
    depression_class_dict = torch.load(config['depression_class_parameter_root'])
    depression_class_net.load_state_dict(depression_class_dict)
    control_cA_net = ConvolutionNet(config)
    control_cA_dict = torch.load(config['control_cA_parameter_root'])
    control_cA_net.load_state_dict(control_cA_dict)
    control_cD_net = ConvolutionNet(config)
    control_cD_dict = torch.load(config['control_cD_parameter_root'])
    control_cD_net.load_state_dict(control_cD_dict)
    control_class_net = ClassificationNet(config)
    control_class_dict = torch.load(config['control_class_parameter_root'])
    control_class_net.load_state_dict(control_class_dict)
    depression_cA_net.to(device)
    depression_cD_net.to(device)
    depression_class_net.to(device)
    control_cA_net.to(device)
    control_cD_net.to(device)
    control_class_net.to(device)
    val_tst(config, depression_cA_net, depression_cD_net, depression_class_net,
            control_cA_net, control_cD_net, control_class_net)