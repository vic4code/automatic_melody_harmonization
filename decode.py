from tonal import pianoroll2number, joint_prob2pianoroll96
import numpy as np
from pypianoroll import Multitrack, Track
import pypianoroll as pr
from matplotlib import pyplot as plt
import os 

# Append argmax index to get pianoroll array
#[batch, beats = 272, chordtypes = 96]
def argmax2pianoroll(joint_prob):
    chord_pianoroll = []
    for song in joint_prob:
        pianoroll = []
        for beat in song:
            pianoroll.append(joint_prob2pianoroll96(beat))
        chord_pianoroll.append(pianoroll)

    chord_pianoroll = np.asarray(chord_pianoroll)

    accompany_pianoroll = chord_pianoroll * 100
    print(chord_pianoroll.shape)
    return accompany_pianoroll

# augment chord into frame base
def sequence2frame(accompany_pianoroll, chord_groundtruth, beat_resolution=24, beat_per_chord=2):
    print('augment chord into frame base...')
    accompany_pianoroll_frame = []
    chord_groundtruth_frame = []
    for acc_song, truth_song in zip(accompany_pianoroll, chord_groundtruth):
        acc_pianoroll = []
        truth_pianoroll = []
        for acc_beat, truth_beat in zip(acc_song, truth_song):
            for i in range(beat_resolution*beat_per_chord):
                acc_pianoroll.append(acc_beat)
                truth_pianoroll.append(truth_beat)
        accompany_pianoroll_frame.append(acc_pianoroll)
        chord_groundtruth_frame.append(truth_pianoroll)

    accompany_pianoroll_frame = np.asarray(accompany_pianoroll_frame).astype(int)
    chord_groundtruth_frame = np.asarray(chord_groundtruth_frame)
    print('accompany_pianoroll shape:', accompany_pianoroll_frame.shape)
    print('groundtruth_pianoroll shape:', chord_groundtruth_frame.shape)
    return accompany_pianoroll_frame, chord_groundtruth_frame

# write pianoroll
def write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats, beat_resolution=24, beat_per_chord=2):

    print('write pianoroll...')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    counter = 0
    for melody_roll, chord_roll, truth_roll, l, tempo, downbeat in zip(melody_data, accompany_pianoroll_frame,
                                                                            chord_groundtruth_frame, length, tempos,
                                                                            downbeats):
        
        melody_roll, chord_roll, truth_roll = melody_roll[:l], chord_roll[:l], truth_roll[:l]

        track1 = Track(pianoroll=melody_roll)
        track2 = Track(pianoroll=chord_roll)
        track3 = Track(pianoroll=truth_roll)

        generate = Multitrack(tracks=[track1, track2], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)
        truth = Multitrack(tracks=[track1, track3], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)

        pr.write(generate, result_dir + '/generate_' + str(counter) + '.mid')
        pr.write(truth, result_dir + '/groundtruth_' + str(counter) + '.mid')

        fig, axs = generate.plot()
        plt.savefig(result_dir + '/generate_' + str(counter) + '.png')
        plt.close()
        fig, axs = truth.plot()
        plt.savefig(result_dir + '/groundtruth_' + str(counter) + '.png')
        plt.close()

        counter += 1
    
    print('Finished!')
    

# write one pianoroll at once
def write_one_pianoroll(result_dir, filename, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats, beat_resolution=24, beat_per_chord=2):

    print('write pianoroll...')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

        melody_roll, chord_roll, truth_roll = melody_data[:l], accompany_pianoroll_frame[:l], chord_groundtruth_frame[:l]

        track1 = Track(pianoroll=melody_roll)
        track2 = Track(pianoroll=chord_roll)
        track3 = Track(pianoroll=truth_roll)

        generate = Multitrack(tracks=[track1, track2], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)
        truth = Multitrack(tracks=[track1, track3], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)

        pr.write(generate, result_dir + '/' + filename + '.mid')
        pr.write(truth, result_dir + '/' + filename + '.mid')

        fig, axs = generate.plot()
        plt.savefig(result_dir + '/' + filename + '.png')
        plt.close()
        fig, axs = truth.plot()
        plt.savefig(result_dir + '/' + filename + '.png')
        plt.close()
    
    print('Finished!')