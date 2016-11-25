import os, random
import cPickle as pickle
import signal
import model

from midi_to_statematrix import *
from data import *
from multi_training import *

def signal_handler(signame, sf):
    stopflag[0] = True

if __name__ == "__main__":
    path = "../data/"
    mood_dirs = ["sad", "happy", "anxious"]
    music_model_size = [300,300]
    pitch_model_size = [100, 50]
    dropout = 0.5
    epochs = 5
    m = model.Model(music_model_size, pitch_model_size, dropout)
    print("Model initialized.")
    
    sad_pieces = loadPieces(path + mood_dirs[0] + '/music/')
    happy_pieces = loadPieces(path + mood_dirs[1] + '/music/')
    anxious_pieces = loadPieces(path + mood_dirs[2] + '/music/')
    print("Music loaded.")
    music_dict = {"sad": sad_pieces, "happy": happy_pieces, "anxious": anxious_pieces}

    # load image features
    img_features_large = pickle.load(open("img_features_large.p", "rb"))
    
    # 'sample' images
    sad_img = img_features_large["../data/sad/images/1.jpg"]
    happy_img = img_features_large["../data/happy/images/224.jpg"]
    anxious_img = img_features_large["../data/anxious/images/27.jpg"]
    print("Sample image features retrieved.")
    
    # Add all images and their corresponding mood in a dictionary
    img_music_dict = {}
    for img in img_features_large.keys():
        mood = img.split("/")[2]
        pcs_in, pcs_out = getPieceBatch(music_dict[mood])
        img_music_dict[img] = [pcs_in, pcs_out]
    print("All image-music loaded in dictionary.")

    # Load previously saved configurations
    learned_list = pickle.load(open('output/params2300.p'))
    m.learned_config = learned_list
    
    # Train
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    counter = 2301;

    for i in range(epochs):
        for img, music in img_music_dict.iteritems():
            img_feature = img_features_large[img]
            error = m.update_fun(music[0], music[1], img_feature)
            if counter % 100 == 0:
                print "counter {}, error={}".format(counter,error)
                # sad
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["sad"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], sad_img)), axis=0),'output/sample-sad{}'.format(counter))
                
                # happy
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["happy"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], happy_img)), axis=0),'output/sample-happy{}'.format(counter))
                
                #anxious
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["anxious"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], anxious_img)), axis=0),'output/sample-anxious{}'.format(counter))
                
                pickle.dump(m.learned_config,open('output/params{}.p'.format(counter), 'wb'))
            
            counter += 1
            
    signal.signal(signal.SIGINT, old_handler)       

