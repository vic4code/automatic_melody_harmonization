import os

class Constants():
    
    # Training parameters
    TEACHER_FORCING = True
    # Chord length 
    BEAT_PER_CHORD = 2
    CHORDS_PER_BAR = 2
    NUM_CHORDS = 96
    # Beat resolution
    BEAT_RESOLUTION = 24
    # Max chord sequence
    MAX_SEQUENCE_LENGTH = 272