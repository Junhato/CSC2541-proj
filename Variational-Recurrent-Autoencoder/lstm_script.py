import os
import numpy as np
from LSTMVRAE import LSTMVRAE

from midi import utils

if __name__ == "__main__":
    music_dir = os.listdir("music")
    min_length = 71
    data = np.empty([71,88,69])
    
    counter = 0
    for music in music_dir:
        music_path = "music/" + music
        midiread = utils.midiread(music_path)
        if midiread.piano_roll.shape[0] > min_length:
            min_index = max(0, (midiread.piano_roll.shape[0]/2) - (min_length/2) - 1)
            max_index = min((midiread.piano_roll.shape[0]/2) + (min_length/2), midiread.piano_roll.shape[0])
            piano_roll = midiread.piano_roll[min_index:max_index,:] 
        data[:,:,counter] = piano_roll
        counter += 1

    print("loaded music")

