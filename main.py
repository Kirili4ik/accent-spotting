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
from audio_utils import seed_torch, count_parameters
from model import make_config, ClassificationNet
from load_data import make_loaders
from train_eval import train
from augs import ComposeAugs, Volume, Faded, PitchShift, Noizer

LABELS_PATH = 'speakers_cleaned.csv'
DATA_PATH = 'data/recordings/recordings'

def main():
	SEED = 1992
	config = make_config()
	seed_torch(SEED)

	transform = ComposeAugs([Volume(p=0.25), Faded(p=0.25), PitchShift(p=0.25), Noizer(p=0.2)], stretch_p=0.25) 
	train_loader, val_loader = make_loaders(DATA_PATH, LABELS_PATH, transform=transform, bs=128)

	wandb.init(project='dla project', name='small quartznet', config=config)
	model = ClassificationNet(config).to(config['device']) 
	print(f'total params: {count_parameters(model)}')
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	train(config, model, optimizer, train_loader, val_loader)


if __name__ == '__main__':
	main()
