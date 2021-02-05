import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod

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
    
# Parameterized Dataset
class Parameterized_Dataset(Dataset):
    def __init__(self, melody, chord, length, chord_onehot, pitch_pattern_ratio):
#         super().__init__(melody, chord, length, chord_onehot, pitch_pattern_ratio, pitch_pattern_rhythm)
        self.melody = melody
        self.chord = chord
        # (batch,1) -> (batch,1,1)
        self.length = np.expand_dims(length, axis=1)
        self.chord_onehot = chord_onehot
        # (batch,1) -> (batch,1,1)
        self.pitch_pattern_ratio = np.expand_dims(pitch_pattern_ratio, axis=1)
#         self.pitch_pattern_rhythm = np.expand_dims(pitch_pattern_rhythm, axis=1)

    def __getitem__(self, index):
        x = torch.from_numpy(self.melody[index]).float()
        y = torch.from_numpy(self.chord[index]).float()
        l = torch.from_numpy(self.length[index])
        x2 = torch.from_numpy(self.chord_onehot[index]).float()
        r_pitch = torch.from_numpy(self.pitch_pattern_ratio[index]).float()
#         r_rhythm = torch.from_numpy(self.pitch_pattern_rhythm[index]).float()
        
        return x, y, l, x2, r_pitch

    def __len__(self):
        return (self.melody.shape[0])