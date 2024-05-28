import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import maxabs_scale
import os


class GetData:
    def __init__(self, path, signal_size=4096):
        # data
        data_classes = [
            d
            for d in os.listdir(os.path.join(path, "signal"))
            if os.path.isfile(os.path.join(path, "signal", d))
        ]
        data_classes.sort(key=lambda x: int(x[0:-4]))
        data = []
        for file_path in data_classes:
            data.append(np.loadtxt(os.path.join(path, "signal", file_path)))

        data = np.array(data)
        data = np.pad(
            data,
            ((0, 0), (0, signal_size - data.shape[-1])),
            "constant",
            constant_values=(0, 0),
        )
        data = maxabs_scale(data, axis=1)
        self.data = data[:, np.newaxis, :]

        # label
        label_classes = [
            d
            for d in os.listdir(os.path.join(path, "label"))
            if os.path.isfile(os.path.join(path, "label", d))
        ]
        label_classes.sort(key=lambda x: int(x[0:-4]))
        label = []
        for file_path in data_classes:
            label.append(np.loadtxt(os.path.join(path, "label", file_path)))
        label = np.array(label)
        label -= 1
        self.label = label
        # self.label = np.eye(4096)[[int(x) for x in label]]

        # label_cls
        self.label_cls = torch.ones(label.shape[0], 1)

    def get_data(self):
        return self.data, self.label, self.label_cls


class GetDataset(Dataset):
    def __init__(self, data, label, label_cls):
        self.data = data
        self.label = label
        self.label_cls = label_cls

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        label_cls = self.label_cls[index]
        return torch.Tensor(data), label, torch.Tensor(label_cls)

    def __len__(self):
        return len(self.data)
