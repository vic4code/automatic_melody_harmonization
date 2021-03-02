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
from model.VAE import VAE
from scheduler import TeacherForcingScheduler, ParameterScheduler
from sklearn.metrics import accuracy_score

class TrainingVAE():
    def __init__(self, args, step=0, k=0.0025, x0=2500):
        
        self.batch_size = args.batch_size
        self.val_size = args.val_size
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.cuda = args.cuda
        self.step = step
        self.k = k
        self.x0 = x0
        self.training_loss = 0
        self.validation_loss = 0
        self.save_model = args.save_model
        
        tf_rates = [(0.5, 0)]
        tfr_scheduler = TeacherForcingScheduler(*tf_rates[0])
        params_dic = dict(tfr=tfr_scheduler)
        self.param_scheduler = ParameterScheduler(**params_dic)
        
    ## Loss function
    def loss_fn(self,loss_function, logp, target, length, mean, log_var, anneal_function, step, k, x0):

        # Negative Log Likelihood
        NLL_loss = loss_function(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    ## Annealing function 
    def kl_anneal_function(self,anneal_function, step, k, x0):
            if anneal_function == 'logistic':
                return float(1/(1+np.exp(-k*(step-x0))))
            elif anneal_function == 'linear':
                return min(1, step/x0) 

    def load_data(self):
        
        batch_size = self.batch_size
        val_size = self.val_size
        
        # Load data
        print('loading data...')
        melody = np.load('./data/melody_baseline.npy')
        chord = np.load('./data/number_96.npy')
        chord_onehot = np.load('./data/onehot_96.npy')
        length = np.load('./data/length.npy')
    #     weight_chord = np.load('./data/weight_chord_10000.npy')

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
    #     weight_chord = torch.from_numpy(weight_chord).float().to(device)

        # Create dataloader
        print('creating dataloader...')
        dataset = ChordGenerDataset(train_melody, train_chord, train_length, train_chord_onehot)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

        return dataloader, val_chord_onehot, val_length
    
    ## Reconstruction rate (accuracy):
    def cal_reconstruction_rate(self,y_true,y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        acc = accuracy_score(y_true,y_pred)
        print('Accuracy:' + f'{acc:.4f}')
    
    def train(self,device,model,optimizer,dataloader,step,k,x0,loss_function):  
        ########## Training mode ###########
            model.train()
            training_loss = self.training_loss
            self.param_scheduler.train()
            dataloader = dataloader
            
            for _, _, length, chord_onehot in dataloader:

                # melody (512, 272, 12 * 24 * 2)
                # chord (512, 272, 1) 
                # length (512,1)
                # chord_onehot (512, 272, 96)
        
                tfr = self.param_scheduler.step()
                length, chord_onehot = length.to(device).squeeze(), chord_onehot.to(device)
                optimizer.zero_grad()

                # Model prediction
        #         print(chord_onehot.shape)
        #         print(length.shape)
                pred, logp ,mu, log_var, _ = model(chord_onehot,length,**tfr)

                # Arrange 
                pred_flatten = []
                groundtruth_flatten = []
                logp_flatten = []
                length = length.squeeze()

                for i in range(self.batch_size):

                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                    logp_flatten.append(logp[i][:length[i]])

                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,12 * 24 * 2)
                    pred_flatten.append(pred[i][:length[i]])

                    # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                    groundtruth_flatten.append(chord_onehot[i][:length[i]])

                # Rearrange for loss calculatio
                logp_flatten = torch.cat(logp_flatten, dim=0)
                pred_flatten = torch.cat(pred_flatten, dim=0)
                groundtruth_flatten = torch.cat(groundtruth_flatten,dim=0).long()
                groundtruth_index = torch.max(groundtruth_flatten,1).indices

                # loss calculation
                # Cross Entropy
    #             CE = cross_entropy(pred_flatten, groundtruth_index)

                # Add weight to NLL also
                NLL_loss, KL_loss, KL_weight = self.loss_fn(loss_function = loss_function, logp = logp_flatten, target = groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
                self.step += 1
                loss = (NLL_loss + KL_weight * KL_loss) / self.batch_size
                training_loss += loss.item()

                loss.backward()
                optimizer.step()

            print('training_loss: ', training_loss / (17505 // self.batch_size))

    def eval(self,device,model,val_chord_onehot,val_length,step,k,x0,loss_function):
        ########## Evaluation mode ###########
            model.eval()
            validation_loss = self.validation_loss
            self.param_scheduler.eval()
            length, chord_onehot = val_length.to(device).squeeze(), val_chord_onehot.to(device)
            
            tfr = self.param_scheduler.step()
            # Model prediction
            pred, logp ,mu, log_var, _ = model(chord_onehot,length,**tfr)

            # Arrange 
            pred_flatten = []
            groundtruth_flatten = []
            logp_flatten = []
            length = length.squeeze()

            for i in range(self.val_size):
                
                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                logp_flatten.append(logp[i][:length[i]])

                # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                pred_flatten.append(pred[i][:length[i]])

                # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                groundtruth_flatten.append(chord_onehot[i][:length[i]])

            # Rearrange for loss calculatio
            logp_flatten = torch.cat(logp_flatten, dim=0)
            pred_flatten = torch.cat(pred_flatten, dim=0)
            pred_index = torch.max(pred_flatten,1).indices
            groundtruth_flatten = torch.cat(groundtruth_flatten,dim=0).long()
            groundtruth_index = torch.max(groundtruth_flatten,1).indices

            # loss calculation
            # Cross Entropy
    #         CE = cross_entropy(pred_flatten, groundtruth_index)

            # Add weight to NLL also
            NLL_loss, KL_loss, KL_weight = self.loss_fn(loss_function = loss_function, logp = logp_flatten, target = groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
            loss = (NLL_loss + KL_weight * KL_loss) / self.val_size
            validation_loss += loss.item()

            print('validation_loss: ', validation_loss)
            self.cal_reconstruction_rate(groundtruth_index.cpu(),pred_index.cpu())

    ## Model training  
    def run(self):

        batch_size = self.batch_size
        # validation data size
        val_size = self.val_size
        epochs = self.epoch
    #     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        device = torch.device('cuda:' + self.cuda)

        # Load data
        dataloader, val_chord_onehot, val_length = self.load_data()

        # Model
        print('building model...')
        model = VAE(device = device).to(device)
        print(model)

        # Training parameters
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        lambda1 = lambda epoch: 0.995 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        loss_function = torch.nn.NLLLoss(reduction='sum')
    #     cross_entropy = nn.CrossEntropyLoss(weight=weight_chord)

        # Define annealing parameters
        step = self.step
        k = self.k
        x0 = self.x0
        
        print('start training...')
        for epoch in tqdm(range(epochs)):
            print('epoch: ', epoch + 1)
            
            self.train(device,model,optimizer,dataloader,step,k,x0,loss_function)
            self.eval(device,model,val_chord_onehot,val_length,step,k,x0,loss_function)

        # Save recontructed results
        # np.save('reconstructed_one_hot_chords.npy', chord_pred.cpu().detach().numpy()) 

        # Save model
        model_dir = 'output_models/' + self.save_model
        torch.save(model.state_dict(), model_dir + '.pth')

## Main
def main():
    ''' 
    Usage:
    python train.py -save_model trained 
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 

    parser.add_argument('-learning_rate', type=float, default=0.005)   
    parser.add_argument('-val_size', default=500)    
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-save_model', type=str, required=True)
    parser.add_argument('-cuda', type=str, default='0')
    
    args = parser.parse_args()
    
    train = TrainingVAE(args)
    train.run()
    
if __name__ == '__main__':
    main()