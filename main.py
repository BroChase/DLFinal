import dataprocess
import misfunctions
import numpy as np

if __name__ == '__main__':
    load = dataprocess.openimages()
    # Filter out the images to be used for testing and training from the data set.
    # load.load_and_save('train')


    # loads the images from MassTestCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTestResized. All images of size 1100 x 1100 in MassTestResized
    # load.load('Test')
    # loads the images from MassTrainCropped images, normalized pixels 0-255, resized images by padding with 0's and
    # saves them to MassTrainResized. All images of size 1100 x 1100 in MassTrainResized
    # load.load('Train')

    mf = misfunctions.classifications()
    test_y = mf.load_data('Test')
    train_y = mf.load_data('Train')
    print('test')