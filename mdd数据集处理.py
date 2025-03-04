import numpy as np
import math

np.random.seed(42)
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
import random

random.seed(42)
from collections import OrderedDict

from torch.utils.data import Dataset
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import mne


class mdd_dataset(Dataset):
    def __init__(self, data_root, data_label):
        super(mdd_dataset, self).__init__()
        self.root = data_root
        self.label, self.eyes = data_label

    def __getitem__(self, index):
        input_root = self.label + " S" + str(index + 1) + " " + self.eyes + ".edf"
        final_root = os.path.join(self.root, input_root)
        try:
            data = mne.io.read_raw_edf(final_root)
        except IOError:
            try:
                input_root = self.label + " S" + str(index + 1) + "  " + self.eyes + ".edf"
                final_root = os.path.join(self.root, input_root)
                data = mne.io.read_raw_edf(final_root)
            except IOError:
                "Error: 文件不存在"
                raise RuntimeError('IOError')
        raw = data.copy()
        raw.load_data()
        # new_event = mne.make_fixed_length_events(raw, duration=100)
        # raw.info['bads'].extend(['EEG 23A-23R', 'EEG 24A-24R'])
        raw = raw.filter(l_freq=0.5, h_freq=50)
        raw = raw.notch_filter(freqs=50)
        raw.rename_channels(
            {'EEG Fp1-LE': 'Fp1', 'EEG F3-LE': 'F3', 'EEG C3-LE': 'C3', 'EEG P3-LE': 'P3', 'EEG O1-LE': 'O1',
             'EEG F7-LE': 'F7', 'EEG T3-LE': 'T3', 'EEG T5-LE': 'T5', 'EEG Fz-LE': 'Fz', 'EEG Fp2-LE': 'Fp2',
             'EEG F4-LE': 'F4', 'EEG C4-LE': 'C4', 'EEG P4-LE': 'P4', 'EEG O2-LE': 'O2', 'EEG F8-LE': 'F8',
             'EEG T4-LE': 'T4', 'EEG T6-LE': 'T6', 'EEG Cz-LE': 'Cz', 'EEG Pz-LE': 'Pz', 'EEG A2-A1': 'A1'
             })
        raw.set_montage('standard_1020')
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        ica.fit(raw)
        ica.plot_sources(raw, show_scrollbars=True)
        # ica.apply(raw)
        # data1, _ = raw[:]
        # data1 = torch.tensor(data1)
        # if self.label == "H":
        #     save_path = r"E:\mdd_pt\control"
        # else:
        #     save_path = r"E:\mdd_pt\depression"
        # file_name = self.label + str(index + 1) + ".pt"
        # save_path = os.path.join(save_path, file_name)
        # data_dict = {'data': data1}
        # torch.save(data_dict, save_path)
        return 0

if __name__ == '__main__':
    control_dataset = mdd_dataset(r"E:\mdd_data\MDD_data", data_label=("H", "EC"))
    depression_dataset = mdd_dataset(r"E:\mdd_data\MDD_data", data_label=("MDD", "EC"))
    # depression_index_list = [i for i in range(34) if i != 3 and i != 4 and i != 7 and i != 11 and i != 15]
    # control_index_list = [i for i in range(15,16) if i != 11 and i != 17]
    # print(index_list)
    # data1 = depression_dataset[0]
    # raw = data1.copy()
    # raw.load_data()
    # new_event = mne.make_fixed_length_events(raw, duration=303)
    # raw = raw.filter(l_freq=0.5, h_freq=50)
    # raw = raw.notch_filter(freqs=50)
    # raw.rename_channels({'EEG Fp1-LE': 'Fp1', 'EEG F3-LE': 'F3', 'EEG C3-LE': 'C3', 'EEG P3-LE': 'P3', 'EEG O1-LE': 'O1',
    #                      'EEG F7-LE': 'F7', 'EEG T3-LE': 'T3', 'EEG T5-LE': 'T5', 'EEG Fz-LE': 'Fz', 'EEG Fp2-LE': 'Fp2',
    #                      'EEG F4-LE': 'F4', 'EEG C4-LE': 'C4', 'EEG P4-LE': 'P4', 'EEG O2-LE': 'O2', 'EEG F8-LE': 'F8',
    #                      'EEG T4-LE': 'T4', 'EEG T6-LE': 'T6', 'EEG Cz-LE': 'Cz', 'EEG Pz-LE': 'Pz', 'EEG A2-A1': 'A1'})
    # raw.set_montage('standard_1020')
    # raw.plot_psd(fmin=0, fmax=52, sphere=(0, 0, 0, 0.11))
    # raw.plot()
    # # eog_idx_auto, eog_scores = ica.find_bads_eog(raw)
    # muscle_idx_auto, scores = ica.find_bads_muscle(raw)
    # ica.plot_sources(raw, show_scrollbars=True)
    # # print("Automatically found eog artifact ICA components:", eog_idx_auto)
    # # print("Automatically found muscle artifact ICA components:", muscle_idx_auto, scores)
    # ica.exclude = [0, 1]
    # # ica.plot_overlay(raw, exclude=[0, 1, 2, 3, 6, 8, 9, 10, 12])
    # raw_recons = raw.copy()
    # ica.apply(raw_recons)
    # raw_recons.plot()
    # raw.plot()
    # for index_input in depression_index_list:
    #     _ = depression_dataset[index_input]
    # for index_input in control_index_list:
    #     _ = control_dataset[index_input]
    _ = depression_dataset[0]
