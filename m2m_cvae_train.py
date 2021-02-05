import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataloader import ChordGenerDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from model.m2m_CVAE import CVAE

## Loss function
def loss_fn(loss_function, logp, target, length, mean, log_var, anneal_function, step, k, x0):

    # Negative Log Likelihood
    NLL_loss = loss_function(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

## Annealing function 
def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0) 
        
## Model training  
def train(args):
    
    batch_size = args.batch_size
    epochs = args.epoch
#     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda:0')
    
    # validation data size
    val_size = args.val_size

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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lambda1 = lambda epoch: 0.995 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    loss_function = torch.nn.NLLLoss(reduction='sum')

    # Define annealing parameters
    step = 0
    k = 0.0025
    x0 = 2500

    print('start training...')
    for epoch in tqdm(range(epochs)):
        print('epoch: ', epoch + 1)

        ########## Training mode ###########
        model.train()
        training_loss = 0
        for melody, _, length, chord_onehot in dataloader:

            # melody (512, 272, 12 * 24 * 2)
            # chord (512, 272, 1) 
            # length (512,1)
            # chord_onehot (512, 272, 96)

            melody, length, chord_onehot = melody.to(device), length.to(device).squeeze(), chord_onehot.to(device)
            optimizer.zero_grad()

            # Model prediction
    #         print(chord_onehot.shape)
    #         print(length.shape)
            melody_pred, logp ,mu, log_var, _ = model(chord_onehot,melody,length)

            # Arrange 
            pred_flatten = []
            groundtruth_flatten = []
            logp_flatten = []
            length = length.squeeze()
            
            for i in range(batch_size):

                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                logp_flatten.append(logp[i][:length[i]])

                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,12 * 24 * 2)
                pred_flatten.append(melody_pred[i][:length[i]])

                # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                groundtruth_flatten.append(melody[i][:length[i]])

            # Rearrange for loss calculatio
            logp_flatten = torch.cat(logp_flatten, dim=0)
            pred_flatten = torch.cat(pred_flatten, dim=0)
            groundtruth_flatten = torch.cat(groundtruth_flatten,dim=0).long()
            groundtruth_index = torch.max(groundtruth_flatten,1).indices

#             print(pred_flatten.shape)
#             print(groundtruth_flatten.shape)
#             print(groundtruth_index.shape)

            # loss calculation
            # Cross Entropy
    #         CE = cross_entropy(chord_pred_flatten, chord_groundtruth_index)

            # Add weight to NLL also
            NLL_loss, KL_loss, KL_weight = loss_fn(loss_function = loss_function, logp = logp_flatten, target = groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
            step += 1
            loss = (NLL_loss + KL_weight * KL_loss) / batch_size
            training_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('training_loss: ', training_loss / (17505 // batch_size))

        ########## Evaluation mode ###########
        model.eval()
        validation_loss = 0
        melody, length, chord_onehot = val_melody.to(device), val_length.to(device).squeeze(), val_chord_onehot.to(device)

        # Model prediction
        melody_pred, logp ,mu, log_var, _ = model(chord_onehot,melody,length)

        # Arrange 
        pred_flatten = []
        groundtruth_flatten = []
        logp_flatten = []
        length = length.squeeze()

        for i in range(val_size):

            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
            logp_flatten.append(logp[i][:length[i]])

            # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
            pred_flatten.append(melody_pred[i][:length[i]])

            # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
            groundtruth_flatten.append(melody[i][:length[i]])

        # Rearrange for loss calculatio
        logp_flatten = torch.cat(logp_flatten, dim=0)
        pred_flatten = torch.cat(pred_flatten, dim=0)
        groundtruth_flatten = torch.cat(groundtruth_flatten,dim=0).long()
        groundtruth_index = torch.max(groundtruth_flatten,1).indices

        # loss calculation
        # Cross Entropy
#         CE = cross_entropy(chord_pred_flatten, chord_groundtruth_index)

        # Add weight to NLL also
        NLL_loss, KL_loss, KL_weight = loss_fn(loss_function = loss_function, logp = logp_flatten, target = groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
        step += 1
        loss = (NLL_loss + KL_weight * KL_loss) / val_size
        validation_loss += loss.item()

        print('validation_loss: ', validation_loss)

    # Save recontructed results
    # np.save('reconstructed_one_hot_chords.npy', chord_pred.cpu().detach().numpy()) 

    # Save model
    model_dir = 'output_models/' + args.save_model
    torch.save(model.state_dict(), model_dir + '.pth')

## Main
def main():
    ''' 
    Usage:
    python train.py -save_model trained 
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 

    parser.add_argument('-learning_rate', default=0.005)   
    parser.add_argument('-val_size', default=500)    
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-save_model', type=str, required=True)
    
    args = parser.parse_args()
    
    train(args)
    
if __name__ == '__main__':
    main()