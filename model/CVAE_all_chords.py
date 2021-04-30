import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from constants import Constants, Constants_framewise

class CVAE(nn.Module):
    def __init__(self,
                 encoder_hidden_size = Constants_framewise.ENCODER_HIDDEN_SIZE, 
                 decoder_hidden_size = Constants_framewise.DECODER_HIDDEN_SIZE,
                 latent_size = Constants_framewise.LATENT_SIZE, 
                 encoder_num_layers = Constants_framewise.ENCODER_NUM_LAYER,
                 decoder_num_layers = Constants_framewise.DECODER_NUM_LAYER, 
                 batch_size = 512, 
                 device = 'cpu' ):
        super(CVAE, self).__init__()
        
        self.device = device
        self.latent_size = latent_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = nn.LSTM(input_size=Constants_framewise.ALL_NUM_CHORDS, 
                               hidden_size = encoder_hidden_size , 
                               num_layers=encoder_num_layers,
                               batch_first=True, 
                               bidirectional=True)
        
        # Encoder to latent
        self.encoder_output2mean = nn.Linear(encoder_hidden_size * 2, latent_size)
        self.encoder_output2logv = nn.Linear(encoder_hidden_size * 2, latent_size)
        
        # Latent to decoder
        self.latent2decoder_input = nn.Linear(latent_size + Constants.BEAT_RESOLUTION * 2 * 12, decoder_hidden_size // 2)
        
        # Decoder
        self.decoder = nn.LSTM(input_size=decoder_hidden_size // 2, 
                               hidden_size =decoder_hidden_size, 
                               num_layers=decoder_num_layers, 
                               batch_first=True, 
                               dropout=0.2, 
                               bidirectional=True)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(decoder_hidden_size * 2,Constants_framewise.ALL_NUM_CHORDS)

    def encode(self, input,length):
        
        # Pack data to encoder
        packed_x = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        encoder_output , (hidden, _) = self.encoder(packed_x)
        
        # Pad back
        encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True, total_length=Constants.MAX_SEQUENCE_LENGTH)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encoder_output2mean(encoder_output)
        log_var = self.encoder_output2logv(encoder_output)

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
        decoder_input = self.latent2decoder_input(z)
        decoder_output, _ = self.decoder(decoder_input)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(decoder_output)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_chord, input_melody, length):
        
        # Encode
        mu, log_var = self.encode(input_chord,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z,input_melody],dim=-1)
        
        # Decode
        output, softmax = self.decode(z)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_chord

