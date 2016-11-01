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
    dr_layer_size = 600
    music_model_size = [300,300]
    pitch_model_size = [100, 50]
    dropout = 0.5
    epochs = 5000

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    cnn = VGG_16('vgg16_weights_th_dim_ordering_th_kernels.h5')
    cnn.compile(optimizer=sgd, loss='categorical_crossentropy')
    print("CNN initialized.")
    
    m = model.Model(music_model_size, pitch_model_size, dropout)
    print("Model initialized.")
    
    sad_pieces = loadPieces(path + mood_dirs[0] + '/music/')
    happy_pieces = loadPieces(path + mood_dirs[1] + '/music/')
    anxious_pieces = loadPieces(path + mood_dirs[2] + '/music/')
    print("Music loaded.")
    img_dict = {}
    music_dict = {"sad": sad_pieces, "happy": happy_pieces, "anxious": anxious_pieces}
    
    # 'sample' images
    sad_img = get_image_features(cnn, path + "sad/images/moulin-rouge-201.jpg", dr_layer_size)[0]
    happy_img = get_image_features(cnn, path + "happy/images/224.jpg", dr_layer_size)[0]
    anxious_img = get_image_features(cnn, path + "anxious/images/lotr2-137.jpg", dr_layer_size)[0]
    print("Sample image features retrieved.")
    
    # Add all images and their corresponding mood in a dictionary
    for mood in mood_dirs:
        img_dir = os.listdir(path + mood + "/images")
        for image in img_dir:
            img_dict[path + mood + "/images/" + image] = mood
    print("All images loaded in dictionary.")        
    
    # Train
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(epochs):
        for img,mood in img_dict.iteritems():
            print(img)
            img_feature = get_image_features(cnn, img, dr_layer_size)
            pcs_in, pcs_out = getPieceBatch(music_dict[mood])
            error = m.update_fun(pcs_in, pcs_out, img_feature[0])
        if i % 100 == 0:
            print "epoch {}, error={}".format(i,error)
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            # sad
            xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["sad"]))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], sad_img)), axis=0),'output/sample{}'.format(i))
            pickle.dump(m.learned_config,open('output/params{}-sad.p'.format(i), 'wb'))
            
            # happy
            xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["happy"]))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], happy_img)), axis=0),'output/sample{}'.format(i))
            pickle.dump(m.learned_config,open('output/params{}-happy.p'.format(i), 'wb'))
            
            #anxious
            xIpt, xOpt = map(numpy.array, getPieceSegment(music_dict["anxious"]))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), m.predict_fun(batch_len, 1, xIpt[0], anxious_img)), axis=0),'output/sample{}'.format(i))
            pickle.dump(m.learned_config,open('output/params{}-anxious.p'.format(i), 'wb'))
            
    signal.signal(signal.SIGINT, old_handler)       

