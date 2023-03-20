import torch
import scipy.io as scio 

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

class HSI_LiDAR_Dataset(Dataset):
    def __init__(self, hsi_path, lidar_path, label_path, split, transform=None):
        if split == 'train':
            self.hsi_data = scio.loadmat(hsi_path)['HSI_TrSet']
            self.lidar_data = scio.loadmat(lidar_path)['LiDAR_TrSet']
            self.label = scio.loadmat(label_path)['TrLabel']
        if split == 'test':
            self.hsi_data = scio.loadmat(hsi_path)['HSI_TeSet']
            self.lidar_data = scio.loadmat(lidar_path)['LiDAR_TeSet']
            self.label = scio.loadmat(label_path)['TeLabel']
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        data_dict = {}
        hsi_data = self.hsi_data[idx]
        lidar_data = self.lidar_data[idx]
        label = self.label[idx] - 1
        data_dict['hsi_data'] = torch.from_numpy(hsi_data).float()
        data_dict['lidar_data'] = torch.from_numpy(lidar_data).float()
        label = torch.from_numpy(label).int().squeeze(0)
        return data_dict, label

class HSI_LiDAR_Patch_Dataset(Dataset):
    def __init__(self, hsi_path, lidar_path, label_path, split, transform=None):
        if split == 'train':# ['HSI_TeSet']
            self.hsi_data = scio.loadmat(hsi_path)['HSI_TrSet']
            self.lidar_data = scio.loadmat(lidar_path)['LiDAR_TrSet']
            self.label = scio.loadmat(label_path)['TrLabel']
        if split == 'test':
            self.hsi_data = scio.loadmat(hsi_path)['HSI_TeSet']
            self.lidar_data = scio.loadmat(lidar_path)['LiDAR_TeSet']
            self.label = scio.loadmat(label_path)['TeLabel']
        self.hsi_patch = (7, 7, 144)
        self.lidar_patch = (7, 7, 21)
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        data_dict = {}
        hsi_data = self.hsi_data[idx]
        lidar_data = self.lidar_data[idx]
        label = self.label[idx] - 1
        data_dict['hsi_data'] = torch.from_numpy(hsi_data).float().reshape(*self.hsi_patch)
        data_dict['lidar_data'] = torch.from_numpy(lidar_data).float().reshape(*self.lidar_patch)
        label = torch.from_numpy(label).int().squeeze(0)
        return data_dict, label
