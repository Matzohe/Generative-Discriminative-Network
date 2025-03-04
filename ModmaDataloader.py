import torch
from torch.utils.data import Dataset
import scipy.io as sio


# 这里导入的MODMA数据集是以.mat文件储存的
class MODMADataLoader(Dataset):
    def __init__(self, root, label):
        super(MODMADataLoader, self).__init__()
        self.root = root
        self.label = label

    def __getitem__(self, index):
        """
        :param index: a tuple of (patient index, time area)
        :return: a tensor of (128 * 2500), data type: torch.float32
        """
        patient_index, time_area = index
        modma_data = sio.loadmat(self.root)
        output_data = torch.tensor(modma_data[self.label][0][patient_index][0][time_area], dtype=torch.float32)
        return output_data
