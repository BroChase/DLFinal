import os
import re
import numpy as np
import pydicom
import shutil
import argparse
import tensorflow as tf
import cv2
import math
import matplotlib.pyplot as plt

class openimages(object):
    def __init__(self):
        # mass train set fold dirs
        self.DIR = 'D:/ddsmcroppedtest/CBIS-DDSM/'
        self.DIR0 = 'D:/ddsmcroppedtrain/CBIS-DDSM/'
        self.DIR1 = 'D:/MassTestCropped'
        self.DIR2 = 'D:/MassTrainCropped'

        self.DIR3 = './MassTestCropped/'
        self.DIR4 = './MassTestResized/'

        self.DIR5 = './MassTrainCropped/'
        self.DIR6 = './MassTrainResized/'

        self.DIR7 = 'D:/croporpadtest/'
        self.DIR8 = 'D:/croporpadtrain/'

        # calc train set fold dirs
        self.CROP0 = 'D:/calcroppedtest/CBIS-DDSM/'
        self.CROP1 = 'D:/calcroppedtrain/CBIS-DDSM/'
        self.CROP2 = './CalTestCropped/'
        self.CROP3 = './CalTrainCropped/'
        self.CROP4 = './CalTestResized/'
        self.CROP5 = './CalTrainResized/'
        # self.CROP6 = 'D:/CalTrainResized/'

        # image saving
        self.TRAIN0 = './xTrain/'
        self.TEST0 = './xTest/'
        self.TRAIN1 = './x0Train/'
        self.TEST1 = './x0Test/'



        pass

    def load_and_save(self, mode):

        if mode == 'Test':
            count = 0
            for file in os.listdir(self.DIR):
                name = file.split('_')
                savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'
                print(savename)
                for f0 in os.listdir(self.DIR + file):
                    for f1 in os.listdir(self.DIR + file + '/' + f0):
                        for f2 in os.listdir(self.DIR + file + '/' + f0 + '/' + f1):
                            f = str(self.DIR) + str(file) + '/' + str(f0) + '/' + str(f1) + '/' + str(f2)
                            q = os.path.getsize(f)
                            # print(f)
                            if q < 4000000:
                                shutil.copy2(f, self.DIR1 + '/' + savename)
                                count = count + 1
                            savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '.dcm'
        if mode == 'Train':
            count = 0
            for file in os.listdir(self.DIR0):
                name = file.split('_')
                savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'
                print(savename)
                for f0 in os.listdir(self.DIR0 + file):
                    for f1 in os.listdir(self.DIR0 + file + '/' + f0):
                        for f2 in os.listdir(self.DIR0 + file + '/' + f0 + '/' + f1):
                            f = str(self.DIR0) + str(file) + '/' + str(f0) + '/' + str(f1) + '/' + str(f2)
                            q = os.path.getsize(f)
                            # print(f)
                            if q < 4000000:
                                shutil.copy2(f, self.DIR2 + '/' + savename)
                                count = count + 1
                            savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'

    def load_and_save_calc(self, mode):

        if mode == 'Test':
            count = 0
            for file in os.listdir(self.CROP0):
                name = file.split('_')
                savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'
                print(savename)
                for f0 in os.listdir(self.CROP0 + file):
                    for f1 in os.listdir(self.CROP0 + file + '/' + f0):
                        for f2 in os.listdir(self.CROP0 + file + '/' + f0 + '/' + f1):
                            f = str(self.CROP0) + str(file) + '/' + str(f0) + '/' + str(f1) + '/' + str(f2)
                            q = os.path.getsize(f)
                            # print(f)
                            if f2 == '000000.dcm':
                                shutil.copy2(f, self.CROP2 + '/' + savename)
                                count = count + 1
                            savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'
        if mode == 'Train':
            count = 0
            for file in os.listdir(self.CROP1):
                name = file.split('_')
                savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'
                print(savename)
                for f0 in os.listdir(self.CROP1 + file):
                    for f1 in os.listdir(self.CROP1 + file + '/' + f0):
                        for f2 in os.listdir(self.CROP1 + file + '/' + f0 + '/' + f1):
                            f = str(self.CROP1) + str(file) + '/' + str(f0) + '/' + str(f1) + '/' + str(f2)
                            q = os.path.getsize(f)
                            # print(f)
                            if f2 == '000000.dcm':
                                shutil.copy2(f, self.CROP3 + '/' + savename)
                                count = count + 1
                            savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4] + '_' + str(count) + '.dcm'



    def check_images(self, parser, image_dir):
        if not os.path.isdir(image_dir):
            parser.error('Directory Does not exist')
            return image_dir

    def load(self, mode):

        if mode == 'Test':
            for file in os.listdir(self.DIR3):
                ds = pydicom.dcmread(self.DIR3 + file)
                # largest image in data set
                # ds = pydicom.dcmread(self.DIR5 + '/P_00990_RIGHT_CC_686.dcm')
                image = ds.pixel_array
                # rescale the values of the images to 0 - 255 attempting to keep lossless with unit16
                rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                # Checking image is still correct
                # img = Image.fromarray(rescaled)
                # img.show()
                # largets image in data set was found to be 1082 x 1086. added a few hundred pixels to accompany the
                # chance that there are images of longer width vs height dims
                padded_image = self.pad_image(rescaled, 1100, 1100)
                ds.PixelData = padded_image.tobytes()
                ds.Rows, ds.Columns = padded_image.shape
                ds.save_as(self.DIR4 + file)
        elif mode == 'Train':
            for file in os.listdir(self.DIR5):
                ds = pydicom.dcmread(self.DIR5 + file)
                image = ds.pixel_array
                rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                padded_image = self.pad_image(rescaled, 1100, 1100)
                ds.PixelData = padded_image.tobytes()
                ds.Rows, ds.Columns = padded_image.shape
                ds.save_as(self.DIR6 + file)


    def load2(self, mode):
        sess = tf.Session()
        with sess.as_default():
            if mode == 'Test':
                for file in os.listdir(self.DIR3):
                    ds = pydicom.dcmread(self.DIR3 + file)
                    # largest image in data set
                    # ds = pydicom.dcmread(self.DIR5 + '/P_00990_RIGHT_CC_686.dcm')
                    image = ds.pixel_array
                    # rescale the values of the images to 0 - 255 attempting to keep lossless with unit16
                    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                    rescaled = rescaled[..., np.newaxis]
                    # Checking image is still correct
                    # img = Image.fromarray(rescaled)
                    # img.show()
                    # largets image in data set was found to be 1082 x 1086. added a few hundred pixels to accompany the
                    # chance that there are images of longer width vs height dims
                    padded_image = tf.image.resize_image_with_crop_or_pad(rescaled, 512, 512)
                    arr = padded_image.eval()
                    arr = np.squeeze(arr)
                    ds.PixelData = arr.tobytes()
                    ds.Rows, ds.Columns = arr.shape
                    ds.save_as(self.DIR4 + file)  #4
            elif mode == 'Train':
                for file in os.listdir(self.DIR5):
                    ds = pydicom.dcmread(self.DIR5 + file)
                    image = ds.pixel_array
                    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                    rescaled = rescaled[..., np.newaxis]
                    padded_image = tf.image.resize_image_with_crop_or_pad(rescaled, 512, 512)
                    arr = padded_image.eval()
                    arr = np.squeeze(arr)
                    ds.PixelData = arr.tobytes()
                    ds.Rows, ds.Columns = arr.shape
                    ds.save_as(self.DIR6 + file)  #6

    def load3(self, mode):
        sess = tf.Session()
        with sess.as_default():
            if mode == 'Test':
                for file in os.listdir(self.CROP2):
                    ds = pydicom.dcmread(self.CROP2 + file)
                    # largest image in data set
                    # ds = pydicom.dcmread(self.DIR5 + '/P_00990_RIGHT_CC_686.dcm')
                    image = ds.pixel_array
                    # rescale the values of the images to 0 - 255 attempting to keep lossless with unit16
                    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                    rescaled = rescaled[..., np.newaxis]
                    # Checking image is still correct
                    # img = Image.fromarray(rescaled)
                    # img.show()
                    # largets image in data set was found to be 1082 x 1086. added a few hundred pixels to accompany the
                    # chance that there are images of longer width vs height dims
                    padded_image = tf.image.resize_image_with_crop_or_pad(rescaled, 512, 512)
                    arr = padded_image.eval()
                    arr = np.squeeze(arr)
                    ds.PixelData = arr.tobytes()
                    ds.Rows, ds.Columns = arr.shape
                    ds.save_as(self.CROP4 + file)  # 4
            elif mode == 'Train':
                for file in os.listdir(self.CROP3):
                    ds = pydicom.dcmread(self.CROP3 + file)
                    # largest image in data set
                    # ds = pydicom.dcmread(self.DIR5 + '/P_00990_RIGHT_CC_686.dcm')
                    image = ds.pixel_array
                    # rescale the values of the images to 0 - 255 attempting to keep lossless with unit16
                    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint16)
                    rescaled = rescaled[..., np.newaxis]
                    # Checking image is still correct
                    # img = Image.fromarray(rescaled)
                    # img.show()
                    # largets image in data set was found to be 1082 x 1086. added a few hundred pixels to accompany the
                    # chance that there are images of longer width vs height dims
                    padded_image = tf.image.resize_image_with_crop_or_pad(rescaled, 512, 512)
                    arr = padded_image.eval()
                    arr = np.squeeze(arr)
                    #im = Image.fromarray(arr)
                    #im.show()
                    #im.save(self.CROP5 + file + '.png')
                    ds.Rows, ds.Columns = arr.shape
                    ds.PixelData = arr.tobytes()
                    ds.save_as(self.CROP5 + file)

    def pad_image(self, image, target_h, target_w):
        # shape of hxw of image
        original_shape = image.shape
        # check if the bounds are less then or == to zero
        if target_h - original_shape[0] <= 0 or target_w - original_shape[1] <= 0:
            print('out of bounds')
        # if the size is even of the original image then add equal boarders to top and bottom else offset by 1
        if (target_h - original_shape[0]) % 2 == 0:
            h1 = int((target_h - original_shape[0]) / 2)
            h2 = h1
        else:
            h1 = int(math.ceil((target_h - original_shape[0]) / 2))
            h2 = int(math.floor((target_h - original_shape[0]) / 2))

        if (target_w - original_shape[1]) % 2 == 0:
            w1 = int((target_w - original_shape[1]) / 2)
            w2 = w1
        else:
            w1 = int(math.ceil((target_h - original_shape[1]) / 2))
            w2 = int(math.floor((target_h - original_shape[1]) / 2))

        # add the padding to the image with values of 0's
        pad = cv2.copyMakeBorder(image, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=0)
        # check image with padding
        #img = Image.fromarray(pad)
        #img.show()
        return pad

    def load_padded(self, mode):

        if mode == 'Test':
            images = []
            for file in os.listdir(self.DIR4):
                try:
                    ds = pydicom.dcmread(self.DIR4 + file)
                    # get image numpy array
                    image = ds.pixel_array
                    #images.append(image)
                except:
                    print(file)
                try:
                    file = file.split('.')
                    plt.imsave(self.TEST0+file[0]+'.png', image)
                except:
                    print('failed to save image')
            # return np.asarray(images)
            # return images

        elif mode == 'Train':
            images = []
            for file in os.listdir(self.DIR6):
                try:
                    ds = pydicom.dcmread(self.DIR6 + file)
                    image = ds.pixel_array
                    #images.append(image)
                except:
                    print(file)
                try:
                    file = file.split('.')
                    plt.imsave(self.TRAIN0+file[0]+'.png', image)
                except:
                    print('failed to save image')
            # return np.asarray(images)
            # return images

    def load_padded2(self, mode):

        if mode == 'Test':
            images = []
            for file in os.listdir(self.CROP4):
                try:
                    ds = pydicom.dcmread(self.CROP4 + file)
                    # get image numpy array
                    image = ds.pixel_array
                    # images.append(image)
                except:
                    print(file)
                try:
                    file = file.split('.')
                    plt.imsave(self.TEST1+file[0]+'.png', image)
                except:
                    print('failed to save image')
            # return np.asarray(images)
            # return images

        elif mode == 'Train':
            images = []
            for file in os.listdir(self.CROP5):
                try:
                    ds = pydicom.dcmread(self.CROP5 + file)
                    image = ds.pixel_array
                    #images.append(image)
                except:
                    print(file)
                try:
                    file = file.split('.')
                    plt.imsave(self.TRAIN1+file[0]+'.png', image)
                except:
                    print('failed to save image')
            # return np.asarray(images)
            #return images