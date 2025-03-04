import mne
import torch
import scipy.io as sio
import os
from wavelet_process import wt_process
import matplotlib.pyplot as plt

data_root = os.path.join(r"G:\data package\MODMA_data", "data_8trial.mat")
control_data = sio.loadmat(data_root)['control'][0][0][0][0] * 0.1
depression_data = sio.loadmat(data_root)['depression'][0][0][0][0] * 0.1
print(type(control_data))
print(control_data.shape)
plt.plot([i for i in range(2500)], control_data[15] - control_data[16], color='r')
plt.plot([i for i in range(2500)], control_data[14] - control_data[15], color='g')
# plt.plot([i for i in range(2500)], control_data[15], color='y')
plt.show()
cA, cD = wt_process(control_data, 'db6')

new_list = []
for i in range(128):
    if i != 127:
        new_list.append(cA[i] - cA[i + 1])
    else:
        new_list.append(cA[127] - cA[0])
new_cA = torch.cat(new_list, dim=0).view(128, -1)
channel_number = [str(i) for i in range(128)]
sfreq = 25
new_info = mne.create_info(channel_number, sfreq)
raw_data = mne.io.RawArray(depression_data, new_info)
anoter_raw_data = mne.io.RawArray(control_data, new_info)
print(raw_data.info)
anoter_raw_data.plot(block=True)
raw_data.plot(block=True)

