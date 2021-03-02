import argparse
from tqdm import tqdm
import numpy as np
import torch
import pickle
from decode import *
from model.VAE import VAE
from metrics import CHE_and_CC, CTD, CTnCTR, PCS, MCTD
from constants import Constants
from sklearn.metrics import accuracy_score

class InferenceVAE():
    def __init__(self,args):
        
        self.model_path = args.model_path
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else 'cpu'
        self.inference_size = args.inference_size
        self.cuda = args.cuda
        self.save_sample = args.save_sample
        self.decode_to_pianoroll = args.decode_to_pianoroll
        
    def load_data(self):
        # Load data
        print('loading data...')
        
        # Fake data
#         melody_data = np.random.randint(128, size = (512, 12056, 128))
#         melody = np.random.randint(128, size = (512, 272, 2 * 12 * 24))
#         chord_groundtruth = np.random.randint(96 ,size = (512, 272, 128))
#         chord_onehot = np.random.randint(96, size = (512, 272, 96))
#         lengths = np.random.randint(1,272, size = (512,))
#         tempos = np.random.randint(1,180, size = (512))
#         downbeats = np.random.randint(1,180, size = (512))
        
        melody_data = np.load('./data/melody_data.npy')
        chord_groundtruth = np.load('./data/chord_groundtruth.npy')
        chord_onehot = np.load('./data/onehot_96.npy')

        melody = np.load('./data/melody_baseline.npy')
        lengths = np.load('./data/length.npy')

        f = open('./data/tempos', 'rb')
        tempos = pickle.load(f)
        f.close()
        f = open('./data/downbeats', 'rb')
        downbeats = pickle.load(f)
        f.close()

        print('splitting testing set...')
        melody_data = melody_data[:self.inference_size]
        chord_groundtruth = chord_groundtruth[:self.inference_size]

        val_chord = torch.from_numpy(chord_onehot[:self.inference_size]).float()
        val_melody = torch.from_numpy(melody[:self.inference_size]).float()
        val_length = torch.from_numpy(lengths[:self.inference_size])

        tempos = tempos[:self.inference_size]
        downbeats = downbeats[:self.inference_size]
        
        return melody_data, val_melody, chord_groundtruth, val_chord, val_length, tempos, downbeats
        
    ## Reconstruction rate (accuracy):
    def cal_reconstruction_rate(self,y_true,y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        acc = accuracy_score(y_true,y_pred)
        print('Accuracy:' + f'{acc:.2f}')

    def load_model(self,model_path):
        # Load model
        print('building model...')
        model = VAE(device = self.device).to(self.device)
        model.load_state_dict(torch.load('output_models/' + model_path + '.pth')) 
        
        return model
    
    def decode2pianoroll(self,val_length, accompany_pianoroll, chord_groundtruth, BEAT_RESOLUTION=Constants.BEAT_RESOLUTION, BEAT_PER_CHORD=Constants.BEAT_PER_CHORD):
        
        # augment chord into frame base
        accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_groundtruth, BEAT_RESOLUTION=Constants.BEAT_RESOLUTION, BEAT_PER_CHORD=Constants.BEAT_PER_CHORD)

        # length into frame base
        length = val_length * Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD

        # write pianoroll
        result_dir = 'results/' + self.save_sample
        write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats)

    ## Model inference
    def run(self):

        melody_data, val_melody, chord_groundtruth, val_chord, val_length, tempos, downbeats = self.load_data()
        val_chord, val_length = val_chord.to(self.device), val_length.to(self.device).squeeze()
        val_length = val_length.cpu().detach().numpy()
        
        model = self.load_model(self.model_path)
        model.eval()  
        
        ########## Inference ###########
        with torch.no_grad():
            preds, _, _, _, _ =  model(val_chord,val_length,tfr=0)

        # Proceed chord decode
        print('proceed chord decode...')
        preds = preds.cpu().detach().numpy()
        joint_prob = preds

        # Append argmax index to get pianoroll array
        accompany_pianoroll = argmax2pianoroll(joint_prob)
        
        print('chord ground truth',chord_groundtruth.shape)
        print('accompany_pianoroll',accompany_pianoroll.shape)
        
        # Calculate accuracy
        self.cal_reconstruction_rate(chord_groundtruth,accompany_pianoroll)
        
        # Decode to pianoroll or not
        if self.decode_to_pianoroll:
            self.decode2pianoroll(val_length, accompany_pianoroll, chord_groundtruth)

## Main
def main():
    ''' 
    Usage:
    python train.py -save_model trained 
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 
    
    parser.add_argument('-inference_size', default=500) 
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-outputdir', type=str, required=True)
    parser.add_argument('-cuda', type=str, default='0')
    parser.add_argument('-decode_to_pianoroll', default=False)
    
    args = parser.parse_args()
    
    inference = InferenceVAE(args)
    inference.run()
    
if __name__ == '__main__':
    main()