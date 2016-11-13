import os
import cPickle as pickle

from keras.optimizers import SGD
from vgg16_keras_th import VGG_16, get_image_features

if __name__ == "__main__":
    path = "../data/"
    mood_dirs = ["sad", "happy", "anxious"]
    dr_layer_size = 600
    dr_layer_size2 = 200    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    cnn = VGG_16('vgg16_weights_th_dim_ordering_th_kernels.h5')
    cnn.compile(optimizer=sgd, loss='categorical_crossentropy')
    print("CNN initialized.")
    
    img_features_large = {}
    img_features_small = {}
    
    for mood in mood_dirs:
        img_dir = os.listdir(path + mood + "/images")
        for image in img_dir:
            img_path = path + mood + "/images/" + image
            if os.path.isfile(img_path):
                img_feature_time = get_image_features(cnn, img_path, dr_layer_size)
                img_feature_pitch = get_image_features(cnn, img_path, dr_layer_size2)
                img_features_large[img_path] = img_feature_time[0]
                img_features_small[img_path] = img_feature_pitch[0]
    print("All image-features loaded.")
    
    pickle.dump(img_features_large, open("img_features_large.p", "wb") )
    pickle.dump(img_features_small, open("img_features_small.p", "wb") )
