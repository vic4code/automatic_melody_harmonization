#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:49:33 2020

@author: victor
"""
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from model.VAE import *

step = 1
batch_size = 5

# Annealing parameter
k = 0.0025
x0 = 2500

model = VAE().to(device)

input_chord = torch.randn(batch_size, 272, 96)
length = torch.Tensor([272, 100, 200, 150, 120]).long()
chord_pred, mu, log_var, input_x = model(input_chord,length)

# flatten tensor to calculate loss
# chord = torch.empty(batch_size, 272, 96 + 1, dtype=torch.long).random_(0, 1)

chord_pred_flatten = []
length = length.squeeze()

for i in range(batch_size):
    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
    chord_pred_flatten.append(chord_pred[i][:length[i]])

    # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
    # chord_flatten.append(chord[i][:length[i]])

# Rearrange for loss calculation
chord_pred_flatten = torch.cat(chord_pred_flatten, dim=0)
chord_groundtruth_index = torch.empty(chord_pred_flatten.shape[0], dtype=torch.long).random_(0, 96)

# loss calculation
# Add weight to NLL also

NLL_loss, KL_loss, KL_weight = loss_fn(logp = chord_pred_flatten, target = chord_groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)

loss = (NLL_loss + KL_weight * KL_loss) / batch_size
print('loss: ',loss.item())




