
import torch
from torch.utils.data import Dataset, DataLoader


class FftWaveletDataset(Dataset):
    # 注意，在数据处理过程中，我们已经将这些数据做了batch的优化，但是，想要知道，是不是对于不同位置的电极，需要单独训练一个网络，希望不要
    # 因为这里可是有128个电极啊，别说，感觉batch size放32有点大了
    # 对于这里的输入数据，其中已经包含了cA和cD的数据信息
    def __init__(self, data_path, device):
        super(FftWaveletDataset, self).__init__()
        self.data_path = data_path
        self.device = device
        self.cA, self.cD = torch.load(self.data_path)

    def __len__(self):
        return self.cA.shape[0]

    def __getitem__(self, index):
        return self.cA[index].to(self.device), self.cD[index].to(self.device)


if __name__ == '__main__':
    new_dataset = FftWaveletDataset(
        data_path=r"G:\data package\MDD_data\wavelet_construct_data\depression_train_data.pt", device='cuda')
    cA, cD = new_dataset[0]
