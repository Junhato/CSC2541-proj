import os, random
import cPickle as pickle
import signal
import model

from midi_to_statematrix import *
from data import *
from multi_training import *

if __name__ == "__main__":
    path = "../data/"
    mood_dirs = ["sad", "happy", "anxious"]
    music_model_size = [300,300]
    pitch_model_size = [100, 50]
    dropout = 0.5
    #epochs = [7300,7800]
    epochs = [4400,5300,5500]


    m = model.Model(music_model_size, pitch_model_size, dropout)
    print("Model initialized.")

    test_pieces = loadPieces(path + 'test/music/')
    print("Music loaded.")
    
    # Load saved image features
    img_features_test = pickle.load(open("img_features_time_test.p", "rb"))

    for i in epochs:
        # Load previously saved configurations

        learned_list = pickle.load(open('output/params' + str(i) + '.p', 'rb'))
        m.learned_config = learned_list

        counter = 0

        for img in img_features_test:
            xIpt, xOpt = map(numpy.array, getPieceSegment(test_pieces))
            # generate test sample
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], img_features_test[img])), axis=0),'output/sample-{}'.format(str(i)+str(counter)))
            counter += 1