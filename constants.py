import os

class Constants():
    # Make it a multiple of the batch size for best (balanced) performance
    samples_per_ground_truth_data_item = 8
    training_validation_split = 0.9
    # Number of Bars
    beat_per_chord = 1
