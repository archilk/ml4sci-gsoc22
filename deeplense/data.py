import torch
from torch.utils.data import Dataset
import numpy as np
import os
import math

from constants import *

class LensDataset(Dataset):
    def __init__(self, memmap_path, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack where shape of memmap is inferred by creating memmap object twice. TODO: Find cleaner way
        self.x = np.memmap(os.path.join(memmap_path, 'images.npy'), dtype='int32', mode='r')
        self.length = self.x.shape[0] // (IMAGE_SIZE[0] * IMAGE_SIZE[1])
        self.x = np.memmap(os.path.join(memmap_path, 'images.npy'), dtype='int32', mode='r',
                           shape=(self.length,)+IMAGE_SIZE)
        self.y = np.load(os.path.join(memmap_path, 'labels.npy'))

        self.min = self.x.min()
        self.range = self.x.max() - self.min

        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img = (self.x[idx] - self.min) / self.range # Standardize
        img = np.expand_dims(img, axis=0) # Add channel axis
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        
        label = self.y[idx]

        return img, label

class WrapperDataset(Dataset):
    def __init__(self, subset, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
