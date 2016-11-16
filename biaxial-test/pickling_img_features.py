import os
import cPickle as pickle

from keras.optimizers import SGD
from vgg16_keras_th import VGG_16, get_image_features

def pickling_img_features(read_path, layer_size, save_name):
    mood_dirs = ["sad", "happy", "anxious"]  
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    cnn = VGG_16('vgg16_weights_th_dim_ordering_th_kernels.h5')
    cnn.compile(optimizer=sgd, loss='categorical_crossentropy')
    print("CNN initialized.")
    
    img_features = {}
    
    for mood in mood_dirs:
        img_dir = os.listdir(read_path + mood + "/images")
        for image in img_dir:
            img_path = read_path + mood + "/images/" + image
            if os.path.isfile(img_path):
                img_feature = get_image_features(cnn, img_path, layer_size)
                img_features[img_path] = img_feature[0]
    print("All image-features loaded.")
    
    pickle.dump(img_features, open("img_features_" + save_name + ".p", "wb") )

if __name__ == "__main__":
    # pickling_img_features("../data/", 600, "time")
    # pickling_img_features("../data/", 200, "pitch")
    pickling_img_features("../data/test/", 600, "time_test")