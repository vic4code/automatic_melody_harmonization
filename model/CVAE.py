import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class CVAE(nn.Module):
    def __init__(self, lstm_dim = 96, fc_dim = 128, chord_size = 96, latent_size = 16,device = 'cpu'):
        super(CVAE, self).__init__()
        
        self.device = device
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = nn.LSTM(input_size = lstm_dim, hidden_size = fc_dim // 2 , num_layers = 2, batch_first = True, dropout=0.2, bidirectional=True)
        
        # Encoder to latent
        self.hidden2mean = nn.Linear(fc_dim, latent_size)
        self.hidden2logv = nn.Linear(fc_dim, latent_size)
        
        # Latent to decoder
        self.latent2hidden = nn.Linear(latent_size + 12 * 24 * 2, fc_dim)
        
        # Decoder
        self.decoder = nn.LSTM(input_size = fc_dim, hidden_size = fc_dim // 2, num_layers = 2, batch_first = True, dropout=0.2, bidirectional=True)
        
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
#         eps = torch.randn(std.shape)
        
        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z
      
    def decode(self, z, length):
        # If cuda
#         if torch.cuda.is_available():
#             z = z.to(self.device)
            
        # Latent to hidden 
        result = self.latent2hidden(z)
        
        # Pack data to decoder
        packed_x = pack_padded_sequence(result, length.cpu(), batch_first=True, enforce_sorted=False)
        packed_x , (ht, ct) = self.decoder(packed_x)
        
        # Pad back
        hidden, _ = pad_packed_sequence(packed_x, batch_first=True, total_length=272)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(result)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_x,melody, length):
        
        # Note
        # 拿 hidden out output , 再把
        # z - > 改丟到 hidden 
        
        # Encode
        mu, log_var = self.encode(input_x,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Add condition 
        z = torch.cat((z,melody), dim=-1)
        
        # Decode
        output,softmax = self.decode(z, length)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_x
    
    def sample(self, length):
        
        latent_sample = torch.randn(1,length, self.latent_size)
        
        # To tensor
        length = torch.Tensor([length]).long()
        sample = self.decode(latent_sample,length)
        
        return F.softmax(sample,dim=-1)
    
    
def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

## loss function
def loss_fn(loss_function, logp, target, length, mean, log_var, anneal_function, step, k, x0):
    
    # Negative Log Likelihood
    NLL_loss = loss_function(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight
 
 