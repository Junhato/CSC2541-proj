import os, random
import cPickle as pickle
import signal
import model

from keras.optimizers import SGD
from midi_to_statematrix import *
from data import *
from multi_training import *
from vgg16_keras_th import VGG_16, get_image_features

def signal_handler(signame, sf):
    stopflag[0] = True

if __name__ == "__main__":
    path = "../data/"
    mood_dirs = ["sad", "happy", "anxious"]
    music_model_size = [300,300]
    pitch_model_size = [100, 50]
    dropout = 0.35
    epochs = 5    
    m = model.Model(music_model_size, pitch_model_size, dropout)
    print("Model initialized.")
    
    sad_pieces = loadPieces(path + mood_dirs[0] + '/music/')
    happy_pieces = loadPieces(path + mood_dirs[1] + '/music/')
    anxious_pieces = loadPieces(path + mood_dirs[2] + '/music/')
    music_dict = {"sad": sad_pieces, "happy": happy_pieces, "anxious": anxious_pieces}
    print("Music loaded.")
    
    # load image features
    img_features_large = pickle.load(open("img_features_large.p", "rb"))
    img_features_small = pickle.load(open("img_features_small.p", "rb"))
    
    # 'sample' images
    sad_img_feature_time = img_features_large["../data/sad/images/1.jpg"]
    happy_img_feature_time = img_features_large["../data/happy/images/224.jpg"]
    anxious_img_feature_time = img_features_large["../data/anxious/images/27.jpg"]
    sad_img_feature_pitch = img_features_small["../data/sad/images/1.jpg"]
    happy_img_feature_pitch = img_features_small["../data/happy/images/224.jpg"]
    anxious_img_feature_pitch = img_features_small["../data/anxious/images/27.jpg"]
    print("Sample image features retrieved.")
    
    # Add all images and their corresponding mood in a dictionary
    img_music_dict = {}
    for mood in mood_dirs:
        img_dir = os.listdir(path + mood + "/images")
        for image in img_dir:
            img_path = path + mood + "/images/" + image
            if os.path.isfile(img_path):
                pcs_in, pcs_out = getPieceBatch(music_dict[mood])
                img_music_dict[img_path] = [pcs_in, pcs_out]
    print("All image-music loaded in dictionary.")
    
    # Load previously saved configurations
    # learned_list = pickle.load(open('output/params7300.p', 'rb'))
    # m.learned_config = learned_list
    
    # Train
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    counter = 0;

    for i in range(epochs):
        for img, music in img_music_dict.iteritems():
            img_feature_time = img_features_large[img]
            img_feature_pitch = img_features_small[img]
            error = m.update_fun(music[0], music[1], img_feature_time, img_feature_pitch)
            if counter % 100 == 0:
                print "counter {}, error={}".format(counter,error)
                # sad
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["sad"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), \
                    m.predict_fun(batch_len, 1, xIpt[0], sad_img_feature_time, sad_img_feature_pitch)), axis=0),'output/sample-sad{}'.format(counter))
                
                # happy
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["happy"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), \
                    m.predict_fun(batch_len, 1, xIpt[0], happy_img_feature_time, happy_img_feature_pitch)), axis=0),'output/sample-happy{}'.format(counter))
                
                #anxious
                xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["anxious"]))
                noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), \
                    m.predict_fun(batch_len, 1, xIpt[0], anxious_img_feature_time, anxious_img_feature_pitch)), axis=0),'output/sample-anxious{}'.format(counter))
                
                pickle.dump(m.learned_config,open('output/params{}.p'.format(counter), 'wb'))
            
            counter += 1
            
    signal.signal(signal.SIGINT, old_handler)       

