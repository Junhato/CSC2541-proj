from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import h5py, cv2, numpy as np


def VGG_16(weights_path=None):

    # slightly modified to extract features only
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, None, None)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

 #  model.add(Flatten())
 #  model.add(Dense(4096, activation='relu'))
 #  model.add(Dropout(0.5))
 #  model.add(Dense(4096, activation='relu'))
 #  model.add(Dropout(0.5))
 #  model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


def get_image_features(img_path, dr_layer_size):

    # image pre-procesing
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pre-trained model
    model = VGG_16('vgg16_weights_th_dim_ordering_th_kernels.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out_vgg16 = model.predict(im)

    input_fbiaxial = out_vgg16.flatten()

    ff_dr = Sequential()
    ff_dr.add(Dense(dr_layer_size, batch_input_shape=(1,len(input_fbiaxial)),init='uniform'))
    input_fbiaxial = ff_dr.predict(np.array([input_fbiaxial]))

    return input_fbiaxial

if __name__ == "__main__":

    result = get_image_features('143.jpg', 300)
    print result.shape
