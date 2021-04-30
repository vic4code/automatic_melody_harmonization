from tonal import pianoroll2number, joint_prob2pianoroll96
from tonal import tonal_centroid, chord482note, chord962note, note2number
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from model.MusicVAE import MusicVAE
from decode import *
import matplotlib.pyplot as plt

torch.cuda.set_device(3)

# Load model
device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
print('building model...')
model = MusicVAE(teacher_forcing = False, eps_i=0, device = device).to(device)
model.load_state_dict(torch.load('output_models/model_musicvae.pth',map_location='cpu'))
print(model)
model.eval()

## Batch inference
torch.manual_seed(0)
z = torch.randn(500,32).to(device)
_, samples = model.decode(z, None)

gen_chord_index = torch.max(samples[20][:length[20]],-1).indices
print('batch chord index',gen_chord_index)

## Single inference
z_ = z[20].unsqueeze(0)
_, sample = model.decode(z_, None)
gen_chord_index = torch.max(sample[0][:length[20]],-1).indices
print('single chord index',gen_chord_index)