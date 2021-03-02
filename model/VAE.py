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
import random

class VAE(nn.Module):
    def __init__(self, input_dim = 96,
                       hidden_size = 1024, 
                       latent_size = 1024, 
                       num_layers = 1, 
                       batch_size = 512, 
                       max_seq_len = 272,
                       device = 'cpu'):
        
        super(VAE, self).__init__()
        
        self.chord_size = input_dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.hidden_size = hidden_size
        self.init_decoder_input = nn.Parameter(torch.rand(self.chord_size))
        
        # Encoder
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Encoder to latent
        self.hidden2mean = nn.Linear(self.hidden_size * 2, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * 2, latent_size)
        
        # Latent to decoder
        self.latent2decoderinput = nn.Linear(latent_size, input_dim)  
        self.latent2hidden = nn.Linear(latent_size, self.hidden_size * 1)  
        
        # Decoder
        self.decoder = nn.GRU(input_size = input_dim + self.chord_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.hidden_size, input_dim)
        
        
    def encode(self,input,length):
    
        # Pack data to encoder
        packed_x = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        packed_x, hidden = self.encoder(packed_x)
        
        # flatten hidden state
        hidden = hidden.transpose_(0, 1).contiguous()
        hidden = hidden.view(self.batch_size, -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.hidden2mean(hidden)
        log_var = self.hidden2logv(hidden)

        return mu, log_var
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
#         eps = torch.randn(std.shape)
        
        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z
      
    def decode(self, z, tfr, c):
        
        z_in = self.latent2decoderinput(z).unsqueeze(1)
        z_hid = self.latent2hidden(z)
        z_hid = z_hid.unsqueeze(0)
            
        token = self.init_decoder_input.repeat(self.batch_size, 1).unsqueeze(1)
        result = []
        
        # Input to decoder 
        for t in range(self.max_seq_len):
            concat = torch.cat([token, z_in], dim=-1)
            output, z_hid = self.decoder(torch.cat([token, z_in], dim=-1), z_hid)
            
            chd = self.outputs2chord(output)
            result.append(chd)
            
            token = torch.zeros(self.batch_size, 1, self.chord_size).to(z.device).float()
            token[torch.arange(0, self.batch_size), 0, chd.max(-1)[-1]] = 1.
            
            teacher_force = random.random() < tfr
            if teacher_force:
                token = c[:, t].unsqueeze(1)
                
        # Reconstruct to one-hot chord
        # result = self.outputs2chord(result)
        # print('onehot_output_shape',result.shape)
        
        result = torch.cat(result, dim=1)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_x, length, tfr):
        
        # Batch size
        self.batch_size, _, _ = input_x.shape
        
        # Encode
        mu, log_var = self.encode(input_x,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        output,softmax = self.decode(z,tfr,input_x)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_x
    
    

