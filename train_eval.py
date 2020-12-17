import wandb
import os
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
import matplotlib.pyplot as plt
from audio_utils import SmoothCrossEntropyLoss, LogMelSpectrogram, upscale_to_wav_len
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train(config, model, optimizer, train_loader, val_loader=None):
    device = config['device']
    epochs = config['n_epochs']
    criterion = SmoothCrossEntropyLoss(smoothing=0.1).to(device)
    featurizer = LogMelSpectrogram(n_mels=config['n_mels']).to(device)
    for epoch in range(epochs):
        train_epoch(model, optimizer, train_loader, featurizer, criterion, device)
        val_epoch(model, val_loader, featurizer, criterion, device, epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            }, 'latest_checkpoint.pt')

        
def train_epoch(model, optimizer, train_loader, to_mels, criterion, device):
    model.train()
    tr_loss, tr_steps = 0, 0
    
    for batch in tqdm(train_loader):
        wav, target = batch[0].to(device), batch[1].to(device)
        mels = to_mels(wav)

        optimizer.zero_grad()
        prediction, attention_vec = model(mels)
        loss = criterion(prediction, target)
        loss.backward()
        
        tr_loss += loss.item()
        tr_steps += 1
        

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()

        wandb.log({'train loss': tr_loss / tr_steps})
            
            
@torch.no_grad()            
def val_epoch(model, val_loader, to_mels, criterion, device, epoch):
    model.eval()
    val_loss = 0
    
    preds, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batch in tqdm(val_loader):    # val_loader
        wav, target = batch[0].to(device), batch[1].to(device)
        wav_len = wav.shape[1]
        mels = to_mels(wav)
 
        prediction, attention_vec = model(mels)
 
        loss = criterion(prediction, target)
        
        val_loss += loss.item()
        preds = torch.cat([preds, torch.argmax(prediction, -1)])
        labels = torch.cat([labels, target])
    
    fig, axes = plt.subplots(2, 1, figsize=(22, 10))
    probs = upscale_to_wav_len(attention_vec)[:wav_len] 
    mask = probs > probs.quantile(0.75)
    sns.lineplot(x=np.arange(wav_len), y=wav.cpu().numpy().squeeze(), hue=mask.detach().cpu().numpy().squeeze())#, ax=axes[0])
    axes[1].plot(probs.detach().cpu().numpy().squeeze())
    plt.title(str(target.squeeze()))
    plt.savefig(f'val_{epoch}.png')
    wandb.log({'mean val loss': val_loss / len(val_loader), 
                'val accuracy': accuracy_score(labels.cpu(), preds.cpu()),
              'val audio for attention': [wandb.Audio(wav.cpu().numpy().squeeze(), sample_rate=16000)]})
              
