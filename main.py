import dataprocess
import os
import tensorflow as tf
import neural_networks
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

    nn = neural_networks.NeuralNetworks()


    print('test')