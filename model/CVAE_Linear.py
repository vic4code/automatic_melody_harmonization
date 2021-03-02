import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import random
from constants import Constants

class CVAE(nn.Module):
    def __init__(self, input_dim = 96, 
                       hidden_size = 64, 
                       latent_size = 16,
                       encoder_num_layers = 2,
                       decoder_num_layers = 1, 
                       batch_size = 512, 
                       max_seq_len = 272, 
                       device = 'cpu'):
        
        super(CVAE, self).__init__()
        
        self.chord_size = input_dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size = self.hidden_size , num_layers=encoder_num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        
        # Encoder to latent
        self.hidden2mean = nn.Linear(self.hidden_size * 2, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * 2, latent_size)
        
        # Latent to decoder
        self.latent2hidden = nn.Linear(latent_size + Constants.BEAT_RESOLUTION * 2 * 12, self.hidden_size * 2)
        
        # Decoder
#         self.decoder = nn.LSTM(input_size=fc_dim, hidden_size = fc_dim // 2, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.hidden_size * 2, self.chord_size)
        
        
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
#         eps = torch.randn(std.shape)
        
        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z
      
    def decode(self, z):
        
        # Latent to hidden 
        result = self.latent2hidden(z)
                
        # Reconstruct to one-hot chord
        result = self.outputs2chord(result)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_x, melody, length):
        
        # Note
        # 拿 hidden out output , 再把
        # z - > 改丟到 hidden 
        
        # Encode
        mu, log_var = self.encode(input_x,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        z = torch.cat((z,melody), dim = -1)
        
        # Decode
        output,softmax = self.decode(z)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax, logp, mu, log_var, input_x

