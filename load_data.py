import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from augs import ComposeAugs


class AccentDataset(Dataset):
    
    def __init__(self, root, lblpath, idx=None, transform=None):
        super().__init__()
        self.root = root
        self.targets = None
        self.transform = None
        meta = pd.read_csv(lblpath)
        if idx is None:
            idx = np.arange(len(meta.index), dtype=int)
        self.files = meta.loc[idx, 'filename'].values
        self.targets = meta.loc[idx, 'target'].values
        if transform is not None:
            self.transform = transform
            
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        mp3, sr = torchaudio.load(filepath)
        if self.transform is not None:
            mp3 = self.transform(mp3)
        target = self.targets[idx]
        return mp3.squeeze(0), target
  

    def __len__(self):
        return len(self.files)


def make_loaders(root, lblpath, transform=None, bs=512, train=True):
    meta = pd.read_csv(lblpath)
    train_idx, val_idx, _, _ = train_test_split(np.arange(meta.shape[0], dtype=int), meta['target'], test_size=0.15,
                                                   stratify = meta['target'])

    train_dataset = AccentDataset(root, lblpath, train_idx, transform=transform)
    val_dataset = AccentDataset(root, lblpath, val_idx, transform=ComposeAugs([], stretch_p=0))
    weights = 1.0 / meta.loc[train_idx, 'target_frequency'].values 
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=0, pin_memory=True, 
                              drop_last=True, sampler=sampler)
    
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    return train_loader, val_loader