import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataloader import ChordGenerDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from model.CVAE import *

batch_size = 512
epochs = 10
device = torch.device('cuda:1')
# validation data size
val_size = 500

# Load data
print('loading data...')
melody = np.load('./melody_baseline.npy')
chord = np.load('./number_96.npy')
chord_onehot = np.load('./onehot_96.npy')
length = np.load('./length.npy')
weight_chord = np.load('./weight_chord.npy')

#Splitting data
print('splitting validation set...')
train_melody = melody[val_size:]
val_melody = torch.from_numpy(melody[:val_size]).float()
train_chord = chord[val_size:]
val_chord = torch.from_numpy(chord[:val_size]).float()
train_chord_onehot = chord_onehot[val_size:]
val_chord_onehot = torch.from_numpy(chord_onehot[:val_size]).float()
train_length = length[val_size:]
val_length = torch.from_numpy(length[:val_size])
weight_chord = torch.from_numpy(weight_chord).float().to(device)

max_chord_sequence = chord.shape[1] # 272

# Create dataloader
print('creating dataloader...')
dataset = ChordGenerDataset(train_melody, train_chord, train_length, train_chord_onehot)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

# Model
print('building model...')
model = CVAE(device = device).to(device)
print(model)

# Training parameters
optimizer = optim.Adam(model.parameters(), lr=0.005)
lambda1 = lambda epoch: 0.995 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
loss_function = torch.nn.NLLLoss(reduction='sum')
cross_entropy = nn.CrossEntropyLoss(weight=weight_chord)

# Define annealing parameters
step = 0
k = 0.0025
x0 = 2500

print('start training...')
for epoch in range(epochs):
    print('epoch: ', epoch + 1)
    
    ########## Training mode ###########
    model.train()
    chord_loss = 0
    for melody, chord, length, chord_onehot in dataloader:
        
        # melody (512, 272, 12 * 24 * 2)
        # chord (512, 272, 128)
        # length (512,1)
        # chord_onehot (512, 272, 96)
        
        melody, chord, length, chord_onehot = melody.to(device), chord.to(device), length.to(device).squeeze(), chord_onehot.to(device)
        optimizer.zero_grad()
    
        # Model prediction
#         print(chord_onehot.shape)
#         print(length.shape)
        chord_pred,logp ,mu, log_var, input_x = model(chord_onehot,melody,length)
        
        # Arrange 
        chord_pred_flatten = []
        chord_flatten = []
        logp_flatten = []
        length = length.squeeze()

        for i in range(batch_size):
            
            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
            logp_flatten.append(logp[i][:length[i]])
            
            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
            chord_pred_flatten.append(chord_pred[i][:length[i]])
            
            # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
            chord_flatten.append(chord_onehot[i][:length[i]])
        
        # Rearrange for loss calculatio
        logp_flatten = torch.cat(logp_flatten, dim=0)
        chord_pred_flatten = torch.cat(chord_pred_flatten, dim=0)
        chord_flatten = torch.cat(chord_flatten,dim=0).long()
        chord_groundtruth_index = torch.max(chord_flatten,1).indices
        
#         print(chord_pred_flatten.shape)
#         print(chord_flatten.shape)
#         print(chord_groundtruth_index.shape)
        
        # loss calculation
        # Cross Entropy
        CE = cross_entropy(chord_pred_flatten, chord_groundtruth_index)
        
        # Add weight to NLL also
        NLL_loss, KL_loss, KL_weight = loss_fn(loss_function = loss_function, logp = logp_flatten, target = chord_groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
        step += 1
        loss = (CE + KL_weight * KL_loss) / batch_size
        chord_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print('chord_loss: ', chord_loss / (17505 // batch_size))
    
    
    ########## Evaluation mode ###########
    model.eval()
    val_chord_loss = 0
    melody, chord, length, chord_onehot = val_melody.to(device), val_chord.to(device), val_length.to(device).squeeze(), val_chord_onehot.to(device)

    # Model prediction
    chord_pred,logp ,mu, log_var, input_x = model(chord_onehot,melody,length)

    # Arrange 
    chord_pred_flatten = []
    chord_flatten = []
    logp_flatten = []
    length = length.squeeze()

    for i in range(val_size):

        # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
        logp_flatten.append(logp[i][:length[i]])

        # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
        chord_pred_flatten.append(chord_pred[i][:length[i]])

        # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
        chord_flatten.append(chord_onehot[i][:length[i]])

    # Rearrange for loss calculatio
    logp_flatten = torch.cat(logp_flatten, dim=0)
    chord_pred_flatten = torch.cat(chord_pred_flatten, dim=0)
    chord_flatten = torch.cat(chord_flatten,dim=0).long()
    chord_groundtruth_index = torch.max(chord_flatten,1).indices

    # loss calculation
    # Cross Entropy
    CE = cross_entropy(chord_pred_flatten, chord_groundtruth_index)

    # Add weight to NLL also
    NLL_loss, KL_loss, KL_weight = loss_fn(loss_function = loss_function, logp = logp_flatten, target = chord_groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
    step += 1
    loss = (CE + KL_weight * KL_loss) / val_size
    val_chord_loss += loss.item()
    
    print('val_chord_loss: ', val_chord_loss)
    
# Save recontructed results
# np.save('reconstructed_one_hot_chords.npy', chord_pred.cpu().detach().numpy()) 

# Save model
torch.save(model.state_dict(), 'output_models/model_cvae_weighting.pth')