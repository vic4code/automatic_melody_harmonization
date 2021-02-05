from tonal import pianoroll2number, joint_prob2pianoroll96
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pypianoroll import Multitrack, Track
import pypianoroll as pr
import pickle
from matplotlib import pyplot as plt
import os
import random
from model.Parameter_CVAE import *
from decode import *

import argparse

def main():
    parser = argparse.ArgumentParser(description='Set configs to midi sampling.')
    parser.add_argument(
        '--device', default='cpu', help='device')
    parser.add_argument(
        '--modeldir',  default = 'model_pitch_pattern_cvae_weighting.pth', type=str, help='model directory')
    parser.add_argument(
        '--outputdir',  default = 'pitch_pattern_cvae_sample_result', type=str, help='output directory')
    parser.add_argument(
        '--seed', default=30, type=str, help='random seed')
    parser.add_argument(
        '--sample_num', default=10, help='sample number')
    parser.add_argument(
        '--pitch_ratio', default=1, help='set pitch repeated pattern')
    parser.add_argument(
        '--rhythm_ratio', default=1, help='set rhythmic repeated pattern')
    args = parser.parse_args()
    
    # Load data
    device = args.device
    val_size = 500

    print('loading data...')
    melody_data = np.load('./melody_data.npy')
    chord_groundtruth = np.load('./chord_groundtruth.npy')
    chord_onehot = np.load('./onehot_96.npy')

    # Reconstructed chords
    # chord_recon = np.load('./reconstructed_one_hot_chords.npy')

    melody = np.load('./melody_baseline.npy')
    lengths = np.load('./length.npy')

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
    model = CVAE(device = device).to(device)
    model.load_state_dict(torch.load('output_models/' + args.modeldir))

    model.eval()
    val_length = val_length.to(device)

    # sample conditions
    latent_size = 16
    pianoroll_frames = 12 * 24 * 2
    
    np.random.seed(args.seed)
    indices = np.random.randint(500, size=args.sample_num)
    print(indices)
    
    for index in indices:
        melody_truth = np.expand_dims(melody_data[index], axis=0)
        chord_truth = np.expand_dims(chord_groundtruth[index], axis=0)
        tempo = [tempos[index]]
        downbeat = [downbeats[index]]

        melody1 = val_melody[index].unsqueeze(dim=0)
        print(val_length.shape)
        inference_length = torch.Tensor([val_length[index]]).long()

        # Sampling
        seed = args.seed
        torch.manual_seed(seed)

        batch_size = 1
        r_pitch = torch.Tensor([float(args.pitch_ratio)])
#         r_rhythm = torch.Tensor([float(args.rhythm_ratio)])

        r_pitch = r_pitch.view(batch_size,1,1).expand(batch_size,272,1).to(device)
#         r_rhythm = r_rhythm.view(batch_size,1,1).expand(batch_size,272,1).to(device)

        latent = torch.rand(1,272,latent_size)

#         z = torch.cat((latent,melody1,r_pitch,r_rhythm), dim=-1)
        z = torch.cat((latent,melody1,r_pitch), dim=-1)

        _, chord_pred = model.decode(z,inference_length)

        ########## Random sampling ###########
        # Proceed chord decode
        print('proceed chord decode...')
        decode_length = inference_length
        joint_prob = chord_pred.cpu().detach().numpy()

        # Append argmax index to get pianoroll array
        accompany_pianoroll = argmax2pianoroll(joint_prob)

        # augment chord into frame base
        beat_resolution = 24
        beat_per_chord = 2

        accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_truth, beat_resolution=beat_resolution, beat_per_chord=beat_per_chord)

        # length into frame base
        decode_length = decode_length * beat_resolution * beat_per_chord

        # write pianoroll
        result_dir = 'results/' + args.outputdir
#         filename = str(index) + '-pitch-' + str(args.pitch_ratio) + '-rhythm-' + str(args.rhythm_ratio)
        filename = str(index) + '-pitch-' + str(args.pitch_ratio)
        print(result_dir)
        print(result_dir + '/' + filename + '.mid')
        write_one_pianoroll(result_dir, filename ,melody_truth, accompany_pianoroll_frame,chord_groundtruth_frame, decode_length, tempo,downbeat)
    
if __name__ == "__main__":
    main()





