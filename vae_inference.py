import numpy as np
import torch
import pickle
from model.VAE import *
from decode import *

device = torch.device('cuda:1')
val_size = 500

# Load data
print('loading data...')
melody_data = np.load('./melody_data.npy')
chord_groundtruth = np.load('./chord_groundtruth.npy')
chord_onehot = np.load('./onehot_96.npy')

# Reconstructed chords
chord_recon = np.load('./reconstructed_one_hot_chords.npy')

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
model = VAE(device = device).to(device)
model.load_state_dict(torch.load('output_models/model_vae_reconstruction.pth'))

model.eval()
chord, length = val_chord.to(device), val_length.to(device)

########## Reconstruction ###########
# Prediction
chord_pred, _, mu, log_var, input_x = model(chord,length)

# Proceed chord decode
print('proceed chord decode...')
chord_pred = chord_pred.cpu().detach().numpy()
length = length.cpu().detach().numpy()
joint_prob = chord_pred

print(joint_prob.shape)

# Append argmax index to get pianoroll array
# accompany_pianoroll = argmax2pianoroll(joint_prob)

# # augment chord into frame base
# beat_resolution = 24
# beat_per_chord = 2

# accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_groundtruth, beat_resolution=beat_resolution, beat_per_chord=beat_per_chord)

# # length into frame base
# length = length * beat_resolution * beat_per_chord

# # write pianoroll
# result_dir = 'results/vae_reconstruction_result'
# write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats)

########## Forward Sample ###########
# # Prediction
# chord_pred, _, mu, log_var, input_x = model(chord,length)

# # Proceed chord decode
# print('proceed chord decode...')
# chord_pred = chord_pred.cpu().detach().numpy()
# length = length.cpu().detach().numpy()
# joint_prob = chord_pred

# # Append argmax index to get pianoroll array
# accompany_pianoroll = argmax2pianoroll(joint_prob)

# # augment chord into frame base
# accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_groundtruth, beat_resolution=beat_resolution, beat_per_chord=beat_per_chord)

# # length into frame base
# length = length * beat_resolution * beat_per_chord

# # write pianoroll
# result_dir = 'results/vae_forward_sample_result'
# write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats)


