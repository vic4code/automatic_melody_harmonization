import argparse
from tqdm import tqdm
import numpy as np
import torch
import pickle
from decode import *
from model.VAE import VAE
from metrics import CHE_and_CC, CTD, CTnCTR, PCS, MCTD

## Calculate objective metrics
def cal_metrics(melody,chord_pred,length,output_file):

    f = open(output_file + '.txt', 'w')
    m = [0 for i in range(6)]
    for i in range(val_size):
        chord_pred_part = chord_pred[i][:length[i]]
        melody_part = melody[i][:length[i]]
    #     print(chord_pred_part.shape)
    #     print(melody_part.shape)

        che, cc = CHE_and_CC(chord_pred_part, chord_num=96)
        ctd = CTD(chord_pred_part, chord_num=96)
        ctnctr = CTnCTR(melody_part, chord_pred_part, chord_num=96)
        pcs = PCS(melody_part, chord_pred_part, chord_num=96)
        mctd = MCTD(melody_part, chord_pred_part, chord_num=96)
        m[0] += che
        m[1] += cc
        m[2] += ctd
        m[3] += ctnctr
        m[4] += pcs
        m[5] += mctd
        f.write(str(che) + " " + str(cc) + " " + str(ctd) + " " + str(ctnctr) + " " + str(pcs) + " " + str(mctd) + '\n')
    f.close()

    print('CHE: ', m[0]/val_size)
    print('CC: ', m[1]/val_size)
    print('CTD: ', m[2]/val_size)
    print('CTnCTR: ', m[3]/val_size)
    print('PCS: ', m[4]/val_size)
    print('MCTD: ', m[5]/val_size)
    
## Model training  
def inference(args):
    
    device = torch.device('cuda:1')
    
    # validation data size
    val_size = args.val_size

    # Load data
    print('loading data...')
    melody_data = np.load('./data/melody_data.npy')
    chord_groundtruth = np.load('./data/chord_groundtruth.npy')
    chord_onehot = np.load('./data/onehot_96.npy')

    melody = np.load('./data/melody_baseline.npy')
    lengths = np.load('./data/length.npy')

    f = open('tempos', 'rb')
    tempos = pickle.load(f)
    f.close()
    f = open('downbeats', 'rb')
    downbeats = pickle.load(f)
    f.close()

    print('splitting testing set...')
    melody_data = melody_data[:val_size]
    chord_groundtruth = chord_groundtruth[:val_size]

    val_chord = torch.from_numpy(chord_onehot[:val_size]).float()
    val_melody = torch.from_numpy(melody[:val_size]).float()
    val_length = torch.from_numpy(lengths[:val_size])

    tempos = tempos[:val_size]
    downbeats = downbeats[:val_size]

    # Load model
    print('building model...')
    model = VAE(device = device).to(device)
    model.load_state_dict(torch.load('output_models/' + args.load_model + '.pth'))
    model.eval()

    val_chord, val_length = val_chord.to(device), val_length.to(device)

    ########## Inference ###########
    chord_preds = torch.empty(0).to(device)

    for i in tqdm(range(val_size)):
        pred, _, _, _, _ =  model(val_chord[i],val_length[i])
        preds = torch.cat((preds,pred), dim=0)

    # Proceed chord decode
    print('proceed chord decode...')
    preds = preds.cpu().detach().numpy()
    joint_prob = preds

    val_length = val_length.cpu().detach().numpy()

    # Append argmax index to get pianoroll array
    accompany_pianoroll = argmax2pianoroll(joint_prob)

    # augment chord into frame base
    beat_resolution = 24
    beat_per_chord = 2

    accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_groundtruth, beat_resolution=beat_resolution, beat_per_chord=beat_per_chord)

    # length into frame base
    length = val_length * beat_resolution * beat_per_chord

    # write pianoroll
    result_dir = 'results/' + args.save_sample
    write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats)

    # cal metrics
#     val_melody = val_melody.cpu().detach().numpy()
#     cal_metrics(val_melody, chord_preds, val_length, args.save_metrics)
    
## Main
def main():
    ''' 
    Usage:
    python train.py -save_model trained 
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 
    
    parser.add_argument('-val_size', default=500) 
    parser.add_argument('-load_model', type=str, required=True)
    parser.add_argument('-save_sample', type=str, required=True)
    parser.add_argument('-save_metrics', type=str)
    
    args = parser.parse_args()
    
    inference(args)
    
if __name__ == '__main__':
    main()