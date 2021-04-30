import os

class Constants():
    
    # Model parameters
    ENCODER_HIDDEN_SIZE = 1024 // 2
    CONDUCTOR_HIDDEN_SIZE = 1024 // 2
    DECODER_HIDDEN_SIZE = 1024 // 2
    PRENET_HIDDEN_SIZE = 1024 // 2
    
    LATENT_SIZE = 512 // 2
    
    ENCODER_NUM_LAYER = 2
    CONDUCTOR_NUM_LAYER = 2
    DECODER_NUM_LAYER = 2
    PRENET_NUM_LAYER = 2
    
    # Training parameters
    TEACHER_FORCING = True
    
    # Chord length 
    BEAT_PER_CHORD = 2
    CHORDS_PER_BAR = 2
    NUM_CHORDS = 96
    ALL_NUM_CHORDS = 633
    BARS_PER_CONDUCTOR = 8

    # Beat resolution
    BEAT_RESOLUTION = 24
    
    # Max chord sequence
    MAX_SEQUENCE_LENGTH = 272
    
class Constants_framewise():
    
    # Model parameters
    ENCODER_HIDDEN_SIZE = 256 * 4
    DECODER_HIDDEN_SIZE = 256 * 4
    
    LATENT_SIZE = 16 * 8
    
    ENCODER_NUM_LAYER = 3
    DECODER_NUM_LAYER = 3
    
    PRENET_SIZES = [256,128,96]
    PRENET_LAYER = 1
    
    # Chord length 
    BEAT_PER_CHORD = 2
    CHORDS_PER_BAR = 2
    NUM_CHORDS = 96
    ALL_NUM_CHORDS = 633
    
    # Beat resolution
    BEAT_RESOLUTION = 24
    # Max chord sequence
    MAX_SEQUENCE_LENGTH = 272