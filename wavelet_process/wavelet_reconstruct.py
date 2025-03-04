import torch
import torch.nn as nn
import pywt
from Config import set_seed

set_seed(42)


def WaveletReconstruct(family, cA_output, cD_output, cA_original, cD_original):
    cA_output = cA_output.to('cpu').numpy()
    cD_output = cD_output.to('cpu').numpy()
    cA_original = cA_original.to('cpu').numpy()
    cD_original = cD_original.to('cpu').numpy()
    all_num = cA_output.shape[0]
    loss_function = nn.MSELoss()
    loss_list = []
    for i in range(all_num):
        reconstruct_data = pywt.idwt(cA_output, cD_output, family)
        original_data = pywt.idwt(cA_original, cD_original, family)
        new_loss = loss_function(torch.from_numpy(reconstruct_data), torch.from_numpy(original_data))
        loss_list.append(new_loss)
    return torch.tensor(loss_list)
