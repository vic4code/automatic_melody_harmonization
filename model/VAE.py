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

class VAE(nn.Module):
    def __init__(self, lstm_dim = 96, fc_dim = 128, latent_size = 16, num_layers = 2, bidirectional = True, device = 'cpu'):
        super(VAE, self).__init__()
        
        self.device = device
        self.hidden_size = fc_dim // 2
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        
        # Encoder
        self.encoder = nn.LSTM(input_size=lstm_dim, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=self.bidirectional)
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        # Encoder to latent
        self.hidden2mean = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        
        # Latent to decoder
        self.latent2decoderinput = nn.Linear(self.latent_size, lstm_dim)  
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size * self.hidden_factor)  
        
        # Decoder
        self.decoder = nn.LSTM(input_size=lstm_dim, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=self.bidirectional)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(fc_dim, lstm_dim)
        
        
    def encode(self,input,length):
        
        batch_size = input.size(0)
        # Pack data to encoder
        packed_x = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        packed_x, (hidden, _) = self.encoder(packed_x)
       
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
    
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
      
    def decode(self, z):
        
        batch_size = z.size(0)

        # Latent to decoder_input, hidden 
        # (batch, latent_size)
        decoder_input = self.latent2decoderinput(z)
        decoder_input = decoder_input.unsqueeze(1).expand(-1,272,-1)

        # (batch, latent_size)
        hidden = self.latent2hidden(z)
        
        if self.bidirectional or self.num_layers > 1:
            # unflatten decoder_input, hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
            
        #(batch, seq_len, input_size)
        result ,(ht, ct) = self.decoder(decoder_input, (hidden, hidden))
       
        # Reconstruct to one-hot chord
        result = self.outputs2chord(result)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_x, length):
        
        # Note
        # 拿 hidden out output , 再把
        # z - > 改丟到 hidden 
        
        # Encode
        mu, log_var = self.encode(input_x,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        output,softmax = self.decode(z)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_x
    
    # def sample(self, length):
        
    #     latent_sample = torch.randn(1,length, self.latent_size)
        
    #     # To tensor
    #     length = torch.Tensor([length]).long()
    #     sample = self.decode(latent_sample,length)
        
    #     return F.softmax(sample,dim=-1)
    

