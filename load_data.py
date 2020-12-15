import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split


class SameSize(nn.Module):
    
    def __init__(self, L=16000):
        super().__init__()
        self.L = L

    def pad_audio(self, samples):
        l = samples.shape[1]
        if l >= self.L: 
            return samples
        else: 
            return F.pad(samples, (0, self.L - l), 'constant', 0)

    def chop_audio(self, samples):
        l = samples.shape[1]
        beg = torch.randint(high=l - self.L + 1, size=(1, 1))
        return samples[:, beg : beg + self.L]

    def forward(self, wav):
        wav = self.pad_audio(wav)
        wav = self.chop_audio(wav)
        return wav

    
class AccentDataset(Dataset):
    
    def __init__(self, root, lblpath, sample_size=16000, transform=None):
        super().__init__()
        self.root = root
        self.targets = None
        self.transform = None
        meta = pd.read_csv(lblpath)
        self.files = meta.filename.values
        self.targets = meta.target.values
        self.sample = SameSize(sample_size)
        if transform is not None:
            self.transform = transform
            
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        mp3, sr = torchaudio.load(filepath)
        mp3 = self.sample(mp3)
        if self.transform is not None:
            mp3 = self.transform(mp3)
        mp3 = mp3.squeeze()
        target = self.targets[idx]
        return mp3, target
  

    def __len__(self):
        return len(self.files)


def make_loader(root, lblpath, transform=None, bs=512, train=True):
    dataset = AccentDataset(root, lblpath, transform=transform)
    meta = pd.read_csv(lblpath)
    weights = torch.ones_like(torch.Tensor(meta.index), dtype=torch.float32)
    if train:
    	weights = 1 / meta.target_frequency
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    loader = DataLoader(dataset, batch_size=bs, num_workers=0, pin_memory=True, 
                              drop_last=True, sampler=sampler)
    return loader