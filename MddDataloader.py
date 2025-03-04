import torch
from torch.utils.data import Dataset
import os


class MddDataLoader(Dataset):
    def __init__(self, data_root, label, device):
        """
        mdd channel name and order information
        这里1-8和10-17是左右脑对称电极
        'EEG Fp1-LE': 'Fp1', 'EEG F3-LE': 'F3', 'EEG C3-LE': 'C3', 'EEG P3-LE': 'P3', 'EEG O1-LE': 'O1',
        'EEG F7-LE': 'F7', 'EEG T3-LE': 'T3', 'EEG T5-LE': 'T5', 'EEG Fz-LE': 'Fz', 'EEG Fp2-LE': 'Fp2',
        'EEG F4-LE': 'F4', 'EEG C4-LE': 'C4', 'EEG P4-LE': 'P4', 'EEG O2-LE': 'O2', 'EEG F8-LE': 'F8',
        'EEG T4-LE': 'T4', 'EEG T6-LE': 'T6', 'EEG Cz-LE': 'Cz', 'EEG Pz-LE': 'Pz', 'EEG A2-A1': 'A1'
        """
        super(MddDataLoader, self).__init__()
        self.label = label
        self.datadir = os.listdir(os.path.join(data_root, label))
        self.data_root = os.path.join(data_root, label)
        self.device = device

    def __getitem__(self, index):
        try:
            number, map_index = index
            data_path = os.path.join(self.data_root, self.datadir[number])
            output_data = torch.load(data_path)['data'].permute(1, 0, 2)
        except IOError:
            raise RuntimeError('IOError')
        return output_data[map_index].to(self.device)

    def __len__(self):
        return len(self.datadir)


if __name__ == '__main__':
    data_root = r"G:\data package\MDD_data\my_mdd"
    depression_dataset = MddDataLoader(data_root, label='depression', device='cpu')
    control_dataset = MddDataLoader(data_root, label='control', device='cpu')
    print(len(depression_dataset))
    print(len(control_dataset))
    # 29 depression 25 control
    min_length = 1000000
    for i in range(29):
        data1 = depression_dataset[(i, 0)]
        if data1.shape[-1] < min_length:
            min_length = data1.shape[-1]
    for i in range(25):
        data1 = control_dataset[(i, 0)]
        if data1.shape[-1] < min_length:
            min_length = data1.shape[-1]
    print(min_length)
