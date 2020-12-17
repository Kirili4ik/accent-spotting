import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import numpy as np
from torch import distributions
import seaborn as sns

import librosa
from IPython import display as display_
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample

class Stretcher(object):
    def __init__(self, range_low=0.8, range_high=1.2, p=0.5):
        self.low = range_low
        self.high = range_high
        self.p=p

    def __call__(self, sample):
        if np.random.rand() >= 1-self.p:
            wav = librosa.effects.time_stretch(sample.numpy(), np.random.uniform(self.low, self.high))
            wav = torch.from_numpy(wav)
        else:
            wav = sample
        return wav


class PitchShift(object):
    def __init__(self, range_low=-3, range_high=3, sr=16000, p=0.5):
        self.low = range_low
        self.high = range_high
        self.sr = sr
        self.p=p

    def __call__(self, sample):
        if np.random.rand() >= 1-self.p:
            wav = librosa.effects.pitch_shift(sample.numpy(), self.sr, np.random.uniform(self.low, self.high))
            wav = torch.from_numpy(wav)
        else:
            wav = sample
        return wav


class Noizer(object):
    def __init__(self, m=0, d=0.0005, p=0.5):
        self.m = m
        self.d = d
        self.p=p

    def __call__(self, sample):
        if np.random.rand() >= 1-self.p:
            noiser = distributions.Normal(self.m, self.d)
            wav = sample + noiser.sample(sample.size())
            wav.clamp_(-1, 1)
        else:
            wav = sample
        return wav

    
class Volume(object):
    def __init__(self, gain_range=(0, 10), gain_type='amplitude', p=0.5):
        self.gain_range=gain_range
        self.gain_type=gain_type
        self.p=p

    def __call__(self, sample):
        if np.random.rand() >= 1-self.p:
            gain = np.random.uniform(low=self.gain_range[0], high=self.gain_range[1])
            vol = torchaudio.transforms.Vol(gain=gain, gain_type=self.gain_type)
            wav = vol(sample)
        else:
            wav = sample
        return wav

    
class Faded(object):
    def __init__(self, max_len=16000, fade_shape = 'linear',  p=0.5):
        self.max_len = max_len
        self.fade_shape = fade_shape
        self.p=p

    def __call__(self, sample):
        if np.random.rand() >= 1-self.p:
            l = np.random.randint(low=0, high=self.max_len)
            r = self.max_len-l
            fd = torchaudio.transforms.Fade(fade_in_len=l, fade_out_len=r, fade_shape=self.fade_shape)
            wav = fd(sample)
        else:
            wav = sample
        return wav


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

    
class ComposeAugs(object):
    def __init__(self, augs_list, cap=3, resample=True, osr=44100, nsr=16000, sec=5, stretch_p=0.5):
        self.augs_list = augs_list
        self.cap=cap
        self.resampler = Resample(orig_freq=osr, new_freq=nsr)
        self.sampler = SameSize(sec * nsr)
        self.wav_stretcher=Stretcher(p=stretch_p)
 
    def __call__(self, wav):
        wav = self.resampler(wav).squeeze(0)
        wav =  self.wav_stretcher(wav)
        wav = self.sampler(wav.unsqueeze(0)).squeeze(0)

        aug_ids = np.arange(len(self.augs_list))
        np.random.shuffle(aug_ids)
        aug_ids = aug_ids[:self.cap]
        np.sort(aug_ids)
        
        for aug_id in aug_ids:
            wav = self.augs_list[aug_id](wav)
        return wav
