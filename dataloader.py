import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Dataset
class ChordGenerDataset(Dataset):
    def __init__(self, melody, chord, length, chord_onehot):
        self.melody = melody
        self.chord = chord
        # (batch,1) -> (batch,1,1)
        self.length = np.expand_dims(length, axis=1)
        self.chord_onehot = chord_onehot

    def __getitem__(self, index):
        x = torch.from_numpy(self.melody[index]).float()
        y = torch.from_numpy(self.chord[index]).float()
        l = torch.from_numpy(self.length[index])
        x2 = torch.from_numpy(self.chord_onehot[index]).float()
        return x, y, l, x2

    def __len__(self):
        return (self.melody.shape[0])