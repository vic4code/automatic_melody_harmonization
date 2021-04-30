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
                 encoder_hidden_size = 512, 
                 conductor_hidden_size = 512,
                 decoder_hidden_size = 512,
                 latent_size = 256, 
                 encoder_num_layers = 2,
                 conductor_num_layers = 2,
                 decoder_num_layers = 2, 
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
        self.encoder_num_layers = encoder_num_layers
        self.conductor_num_layers = conductor_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.latent_size = latent_size

        # data goes into bidirectional encoder
        self.encoder = torch.nn.LSTM(input_size=input_dim,
                                     hidden_size=encoder_hidden_size,
                                     num_layers=encoder_num_layers,
                                     bidirectional=True,
                                     batch_first=True,)

        # Encoder to latent
        self.hidden2mean = nn.Linear(self.encoder_hidden_size * encoder_num_layers * 2, latent_size)
        self.hidden2logv = nn.Linear(self.encoder_hidden_size * encoder_num_layers * 2, latent_size)
        
        # Latent to decoder
        self.latent2conductor_input = nn.Linear(latent_size, input_dim)  
        self.latent2conductor_hidden = nn.Linear(latent_size, conductor_hidden_size)  
        
        self.dropout = nn.Dropout(p=0.2)

        # Define the conductor and note decoder
        self.conductor = nn.LSTM(input_size=input_dim,
                                 hidden_size=conductor_hidden_size,
                                 num_layers=conductor_num_layers,
                                 batch_first=True)
        
        self.decoder = nn.LSTM(input_size=Constants.NUM_CHORDS + conductor_hidden_size,
                               hidden_size=decoder_hidden_size,
                               num_layers=decoder_num_layers,
                               batch_first=True)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.decoder_hidden_size, Constants.NUM_CHORDS)

    # used to initialize the hidden layer of the encoder to zero before every batch
    # nn.LSTM will do this by itself, this might be redunt. Look at: https://discuss.pytorch.org/t/when-to-initialize-lstm-hidden-state/2323/16
    def init_hidden(self, hidden_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, self.batch_size, hidden_size, device=self.device)
        c0 = torch.zeros(2, self.batch_size, hidden_size, device=self.device)

        # 2 because has 2 layers
        # n_layers_conductor
        init_conductor = torch.zeros(self.conductor_num_layers,
                                     self.batch_size,
                                     hidden_size,
                                     device=self.device)
        c_condunctor = torch.zeros(self.conductor_num_layers,
                                   self.batch_size,
                                   hidden_size,
                                   device=self.device)

        return init, c0, init_conductor, c_condunctor

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
        encoder_output,(hidden,c) = self.encoder(packed_x)
        
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
        
        _, _, _, cconductor = self.init_hidden(self.decoder_hidden_size)
        
        z_in = self.latent2conductor_input(z).unsqueeze(1)

        # Check conductor layer numbers
        if self.conductor_num_layers > 1:
            z_hid = self.latent2conductor_hidden(z).unsqueeze(0).repeat(self.conductor_num_layers,1,1)
        
        else:
            z_hid = self.latent2conductor_hidden(z).unsqueeze(0)

        conductor_input = z_in
        conductor_hidden = (z_hid,cconductor)

        token = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS, device=self.device)
        input_chord_seqs = torch.cat([token, input_chord_seqs], dim=1)
        
        counter = 0
        blank_chord = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS, device=self.device)
        output_chord_seqs = torch.zeros(self.batch_size,
                            Constants.MAX_SEQUENCE_LENGTH,
                            Constants.NUM_CHORDS,
                            device=self.device)

        # Go through each element in the latent sequence
        # TODO: z.shape[1] = 512? why for i in range(16), only use the first 16?
        for i in range(Constants.MAX_SEQUENCE_LENGTH // Constants.CHORDS_PER_BAR):
            
            embedding, conductor_hidden = self.conductor(
                conductor_input, conductor_hidden)
            
            if self.use_teacher_forcing():

                # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(self.decoder_num_layers,
                                              self.batch_size,
                                              self.decoder_hidden_size,
                                              device=self.device),
                                  torch.randn(self.decoder_num_layers,
                                              self.batch_size,
                                              self.decoder_hidden_size,
                                              device=self.device))

                embedding = embedding.expand(self.batch_size, Constants.CHORDS_PER_BAR,
                                             embedding.shape[2])
                
                # Concat embedding with the previous chord
                decoder_input = torch.cat([embedding, input_chord_seqs[:, range(i * Constants.CHORDS_PER_BAR, i * Constants.CHORDS_PER_BAR + Constants.CHORDS_PER_BAR), :]],dim=-1)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                chord = self.outputs2chord(decoder_output)
                output_chord_seqs[:, range(i * Constants.CHORDS_PER_BAR, i * Constants.CHORDS_PER_BAR + Constants.CHORDS_PER_BAR), :] = chord
                
            else:
                # Reset the decoder state 
                decoder_hidden = (torch.randn(self.decoder_num_layers,
                                              self.batch_size,
                                              self.decoder_hidden_size,
                                              device=self.device),
                                  torch.randn(self.decoder_num_layers,
                                              self.batch_size,
                                              self.decoder_hidden_size,
                                              device=self.device))

                for _ in range(Constants.CHORDS_PER_BAR):
                    # Concat embedding with previous chord                    
                    decoder_input = torch.cat([embedding, blank_chord], dim=-1)
                    decoder_input = decoder_input.view(self.batch_size, 1, -1)

                    # Generate a single note (for each batch)
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                    chord = self.outputs2chord(decoder_output)
                    output_chord_seqs[:, counter, :] = chord.squeeze()
                    counter = counter + 1
                    
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