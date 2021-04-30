import torch
from torch import nn
from torch.nn.functional import softplus
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from constants import Constants
import numpy as np

class MusicVAE(nn.Module):
    def __init__(self,
                 teacher_forcing, 
                 eps_i,
                 input_dim = 96, 
                 encoder_hidden_size = Constants.ENCODER_HIDDEN_SIZE, 
                 conductor_hidden_size = Constants.CONDUCTOR_HIDDEN_SIZE,
                 decoder_hidden_size = Constants.DECODER_HIDDEN_SIZE,
                 latent_size = Constants.LATENT_SIZE, 
                 encoder_num_layer = Constants.ENCODER_NUM_LAYER,
                 conductor_num_layer = Constants.CONDUCTOR_NUM_LAYER,
                 decoder_num_layer = Constants.DECODER_NUM_LAYER, 
                 batch_size = 512, 
                 device = 'cpu' 
                 ):
        
        super(MusicVAE, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i
        self.encoder_hidden_size = encoder_hidden_size
        self.conductor_hidden_size = conductor_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_num_layer = encoder_num_layer
        self.conductor_num_layer = conductor_num_layer
        self.decoder_num_layer = decoder_num_layer
        self.latent_size = latent_size

        # data goes into bidirectional encoder
        self.encoder = nn.GRU(input_size=input_dim, 
                              hidden_size=encoder_hidden_size, 
                              num_layers=encoder_num_layer, 
                              batch_first=True, 
                              bidirectional=True)

        # Encoder to latent
        self.hidden2mean = nn.Linear(self.encoder_hidden_size * encoder_num_layer * 2, latent_size)
        self.hidden2logv = nn.Linear(self.encoder_hidden_size * encoder_num_layer * 2, latent_size)
        
        # Latent to decoder
        self.latent2conductor_input = nn.Linear(latent_size, input_dim)  
        self.latent2conductor_hidden = nn.Linear(latent_size, conductor_hidden_size)  
        
        self.dropout = nn.Dropout(p=0.2)

        # Define the conductor and note decoder
        self.conductor = nn.GRU(input_size = input_dim, 
                                hidden_size=self.conductor_hidden_size, 
                                num_layers=decoder_num_layer, 
                                batch_first=True, 
                                bidirectional=False)
        
        self.decoder = nn.GRU(input_size = Constants.NUM_CHORDS + self.decoder_hidden_size,
                              hidden_size=decoder_hidden_size, 
                              num_layers=decoder_num_layer, 
                              batch_first=True, 
                              bidirectional=False)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.decoder_hidden_size, Constants.NUM_CHORDS)

    # Coin toss to determine whether to use teacher forcing on a note(Scheduled sampling)
    # Will always be True for eps_i = 1.
    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf

    def set_scheduled_sampling(self, eps_i):
        self.eps_i = 1  #eps_i
    
    def encode(self,input,length):
        # Pack data to encoder
        # encoder_output,(hidden,c) = self.encoder(input)
        packed_x = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        encoder_output,hidden = self.encoder(packed_x)
        
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
      
    def decode(self, z, input_chord_seqs):
        conductor_input = self.latent2conductor_input(z).unsqueeze(1)

        if self.conductor_num_layer > 1:
            conductor_hidden = self.latent2conductor_hidden(z).unsqueeze(0).repeat(self.conductor_num_layer,1,1)
        else:
            conductor_hidden = self.latent2conductor_hidden(z).unsqueeze(0)

#         bar_token = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS, device=self.device)
#         input_chord_seqs = torch.cat([bar_token, input_chord_seqs], dim=1)

        chord_token = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS, device=self.device)
        output_chord_seqs = []
    
        for i in range(Constants.MAX_SEQUENCE_LENGTH // Constants.CHORDS_PER_BAR):
#             embedding, conductor_hidden = self.conductor(torch.cat([bar_token,conductor_input],dim=-1), conductor_hidden)
            embedding, conductor_hidden = self.conductor(conductor_input, conductor_hidden)
            decoder_hidden = conductor_hidden 
            
            if self.use_teacher_forcing():
                # Concat embedding with the previous chord
                embedding = embedding.expand(self.batch_size, Constants.CHORDS_PER_BAR, embedding.shape[2])
                decoder_input = torch.cat([embedding, input_chord_seqs[:, range(i * Constants.CHORDS_PER_BAR, i * Constants.CHORDS_PER_BAR + Constants.CHORDS_PER_BAR), :]],dim=-1)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                chord = self.outputs2chord(decoder_output)
                output_chord_seqs.append(chord)
                
            else:
                for _ in range(Constants.CHORDS_PER_BAR):
                    # Concat embedding with previous chord                    
                    decoder_input = torch.cat([embedding, chord_token], dim=-1)
                    decoder_input = decoder_input.view(self.batch_size, 1, -1)
                    
                    # Generate a single note (for each batch)
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                    chord = self.outputs2chord(decoder_output)
                    output_chord_seqs.append(chord)
                    chord_token = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS).to(self.device).float()
                    chord_token[torch.arange(0, self.batch_size), 0, chord.max(-1)[-1].squeeze()] = 1.

        output_chord_seqs = torch.cat(output_chord_seqs, dim=1)
        # Softmax
        softmax = F.softmax(output_chord_seqs,dim=-1)
        
        return output_chord_seqs, softmax

    def forward(self, input_chord_seqs, length):

        # Batch size
        self.batch_size, _, _ = input_chord_seqs.shape
        
        # Encode
        mu, log_var = self.encode(input_chord_seqs,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        output,softmax = self.decode(z, input_chord_seqs)

        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
        
        return softmax, logp, mu, log_var, input_chord_seqs