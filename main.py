import dataprocess
import os
import tensorflow as tf
import neural_networks
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
print(tf.__version__)

if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    load = dataprocess.openimages()
    # Filter out the images to be used for testing and training from the data set.
    # load.load_and_save('Test')
    # load.load_and_save('Train')

    # Calc train set
    # load.load_and_save_calc('Test')
    # load.load_and_save_calc('Train')

    # loads the images from MassTestCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTestResized. All images of size 1100 x 1100 in MassTestResized
    # load.load('Test')
    # loads the images from MassTrainCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTrainResized. All images of size 1100 x 1100 in MassTrainResized
    # DONT USE load.load('Train')

    # Resize the files in MassTestCropped into MassTestResized
    # load.load2('Test')
    # Resize the files in MassTrainCropped into MassTrainResized
    # load.load2('Train')

    # Resize the files in CalTestCropped into CalTestResized
    # load.load3('Test')
    # Resize the files in CalTrainCropped into CalTrainResized
    #load.load3('Train')

    #load.load_padded('Test')
    #load.load_padded('Train')
    #load.load_padded2('Test')
    #load.load_padded2('Train')

    nn = neural_networks.NeuralNetworks()
    nn.cnn()
    # nn.cnn2()
    print('test')