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

cuda = False
device = torch.device("cuda" if cuda else "cpu")

class VAE(nn.Module):
    def __init__(self, lstm_dim = 96 + 1, fc_dim = 128, chord_size = 96 + 1, latent_size = 16):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size=lstm_dim, hidden_size = fc_dim // 2 , num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        
        # Decoder
        self.decoder = nn.LSTM(input_size=fc_dim, hidden_size = fc_dim // 2, num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        
        # Encoder to latent
        self.hidden2mean = nn.Linear(fc_dim, latent_size)
        self.hidden2logv = nn.Linear(fc_dim, latent_size)
        
        # Latent to decoder
        self.latent2hidden = nn.Linear(latent_size, fc_dim)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(fc_dim, chord_size)
        
        
    def encode(self, input,length):
        
        # Pack data to encoder
        packed_x = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        packed_x , (ht, ct) = self.encoder(packed_x)
        
        # Pad back
        hidden, _ = pad_packed_sequence(packed_x, batch_first=True, total_length=272)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.hidden2mean(hidden)
        log_var = self.hidden2logv(hidden)

        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
      
    def decode(self, z, length):
        
        # Latent to hidden 
        result = self.latent2hidden(z)
        
        # Pack data to decoder
        packed_x = pack_padded_sequence(result, length, batch_first=True, enforce_sorted=False)
        packed_x , (ht, ct) = self.decoder(packed_x)
        
        # Pad back
        hidden, _ = pad_packed_sequence(packed_x, batch_first=True, total_length=272)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(result)
        
        return result
    
    def forward(self, input_x, length):
        
        # Encode
        mu, log_var = self.encode(input_x,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Add Condition
        # z = torch.cat((z,melody) , dim=-1)
        
        # Decode
        output = self.decode(z, length)
        
        # Softmax
        logp = F.log_softmax(output, dim=-1)
        
        return logp, mu, log_var, input_x
    
    def sample(self, length):

        sample = torch.randn(melody_length, 1)
        sample = self.decode(sample)

        return sample
    

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

NLL = torch.nn.NLLLoss(reduction='sum')

## loss function
def loss_fn(logp, target, length, mean, log_var, anneal_function, step, k, x0):
    
    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight
 

if __name__ == "__main__":

    
    step = 1
    batch_size = 5
    
    # Annealing parameter
    k = 0.0025
    x0 = 2500
    
    model = VAE().to(device)
    
    input_chord = torch.randn(batch_size, 272, 96 + 1)
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
    NLL_loss, KL_loss, KL_weight = loss_fn(logp = chord_pred_flatten, target = chord_groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
    
    loss = (NLL_loss + KL_weight * KL_loss) / batch_size
    print('loss: ',loss.item())




