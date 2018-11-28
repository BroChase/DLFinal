import dataprocess
import misfunctions
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
print(tf.__version__)

if __name__ == '__main__':
    os.environ['PATH'] += os.pathsep + 'C/Program Files (x86)/Graphviz2.38/bin/'

    load = dataprocess.openimages()
    # Filter out the images to be used for testing and training from the data set.
    # load.load_and_save('train')


    # loads the images from MassTestCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTestResized. All images of size 1100 x 1100 in MassTestResized
    # load.load('Test')
    # loads the images from MassTrainCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTrainResized. All images of size 1100 x 1100 in MassTrainResized
    # load.load('Train')

    mf = misfunctions.Classifications()
    y_test = mf.load_data('Test')
    y_train = mf.load_data('Train')

    x_test = load.load_padded('Test')
    x_train = load.load_padded('Train')

    min = 0
    max = 255
    for i in range(len(x_test)):
        x_test[i] = (x_test[i] - min) / (max - min)

    for i in range(len(x_train)):
        x_train[i] = (x_train[i] - min) / (max - min)

    x_test = np.asarray(x_test)
    x_train = np.asarray(x_train)

    x_train = x_train.reshape(x_train.shape[0], 1100, 1100, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1100, 1100, 1).astype('float32')

    # number of classes
    num_of_classes = y_test.shape[1]
    num_of_pixels = x_train.shape[1]

    class_names = np.unique(y_test)
    print('test')