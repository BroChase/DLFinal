import misfunctions
import numpy as np
import dataprocess


class NeuralNetworks:
    @staticmethod
    def cnn():
        seed = 686
        np.random.seed(seed)

        load = dataprocess.openimages()
        mf = misfunctions.Classifications()

        y_test = mf.load_data('Test')
        y_train = mf.load_data('Train')

        x_test = load.load_padded('Test')
        x_train = load.load_padded('Train')

        min = np.min(x_test)
        max = np.max(x_test)
        for i in range(len(x_test)):
            x_test[i] = ((x_test[i] - min) / (max - min)).astype('float32')

        for i in range(len(x_train)):
            x_train[i] = ((x_train[i] - min) / (max - min)).astype('float32')

        x_test = np.asarray(x_test)
        x_train = np.asarray(x_train)

        x_train = x_train.reshape(x_train.shape[0], 1100, 1100, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 1100, 1100, 1).astype('float32')

        # number of samples
        num_of_classes = y_test.shape[0]
        # number of pixels
        num_of_pixels = x_train.shape[1]

        class_names = np.unique(y_test)