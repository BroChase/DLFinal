import misfunctions
import numpy as np
import dataprocess
import time
from keras import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras import backend as K

K.set_image_dim_ordering('tf')


class NeuralNetworks:
    @staticmethod
    def cnn():
        seed = 686
        np.random.seed(seed)

        load = dataprocess.openimages()
        mf = misfunctions.Classifications()
        mfeval = misfunctions.Eval()
        # mass data truths
        y_test = mf.load_data_mass('Test')
        y_train = mf.load_data_mass('Train')
        # cal data truths
        y_test2 = mf.load_data_calc('Test')
        y_train2 = mf.load_data_calc('Train')
        # concatenate the mass data truths with the cal data truths for test and train
        y_test = np.concatenate((y_test, y_test2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)


        x_test = load.load_padded('Test') + load.load_padded2('Test')
        x_train = load.load_padded('Train') + load.load_padded2('Train')

        # x_test2 = load.load_padded2('Test')
        # x_train2 = load.load_padded2('Train')



        min = np.min(x_test)
        max = np.max(x_test)
        for i in range(len(x_test)):
            x_test[i] = ((x_test[i] - min) / (max - min)).astype(np.float16)

        for i in range(len(x_train)):
            x_train[i] = ((x_train[i] - min) / (max - min)).astype(np.float16)

        x_test = np.array(x_test, dtype=np.float16)
        x_train = np.asarray(x_train, dtype=np.float16)

        x_train = x_train.reshape(x_train.shape[0], 512, 512, 1).astype(np.float16)
        x_test = x_test.reshape(x_test.shape[0], 512, 512, 1).astype(np.float16)

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # number of samples
        num_of_classes = y_test.shape[1]
        # number of pixels
        num_of_pixels = x_train.shape[1]

        # 0 = benign 1 = malignant
        class_names = np.unique(y_test)

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(num_of_pixels, num_of_pixels, 1),
                         activation='relu', strides=2, padding='valid'))

        #model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='valid'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', strides=2, padding='valid'))

        #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='valid'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        #model.add(Dense(units=1000, activation='relu'))

        model.add(Dense(units=100, activation='relu'))

        model.add(Dense(units=num_of_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        mfeval.model_summary(model, 'cnn.png')

        t = time.localtime(time.time())
        timeStamp = str(t.tm_year) + '-' + str(t.tm_mon)\
                    + '-' + str(t.tm_mday) + '--' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec)

        tBoard = TensorBoard(log_dir='logs/{}'.format(timeStamp))

        num_epochs = 50
        batch_size = 32

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=batch_size,
                            verbose=2, callbacks=[tBoard])

        mfeval.final_eval(model, x_test, y_test, history, class_names, 'cnn')
