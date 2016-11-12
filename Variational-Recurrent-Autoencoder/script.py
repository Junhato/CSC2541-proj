import os
import numpy as np
from VRAE import VRAE

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

    print(data.shape)
    print("loaded music")

    hu_encoder = 400
    hu_decoder = 400
    features = 88
    latent_variables = 20
    b1 = 0.95
    b2 = 0.999
    batch_size = 5
    #batch_size = 100
    n_latent = 20
    #n_epochs = 40
    n_epochs = 10
    learning_rate = 0.001
    lam = 0
    path = "./"

    my_vrae = VRAE(hu_encoder, hu_decoder, features, \
     latent_variables, b1, b2, learning_rate, 0.5, batch_size)

    print("initialized model")

    my_vrae.create_gradientfunctions(data)
    print("created gradientfunctions")

    batch_order = np.arange(int(69 / my_vrae.batch_size))
    epoch = 0
    LB_list = []

    print "iterating"
    while epoch < n_epochs:
        epoch += 1
        np.random.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            batch_LB = my_vrae.updatefunction(batch, epoch)
            LB += batch_LB

        LB /= len(batch_order)
        print(LB)
        LB_list = np.append(LB_list, LB)
        print "Epoch", epoch, "finished."
        np.save(path + "LB_list.npy", LB_list)
        my_vrae.save_parameters(path)
            
    
    #my_z = np.random.normal(0, 1, (latent_variables,1))
    t_steps = 71
    z, mu, sigma = my_vrae.encode(data[:,:,0])
    sample_roll = my_vrae.decode(t_steps, latent_variables, z) 
    print "original piano_roll"
    print(data[:,:,0])
    print "decoded piano_roll"
    print(sample_roll)
    outfile = "sample.mid"
    utils.midiwrite(outfile, sample_roll)
