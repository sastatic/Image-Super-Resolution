#############################
# import all libraries used #
#############################

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy
import math
import os
import cv2

#####################################################
# created model object which is model of our method #
#####################################################
def predict_model():

    # adding 3 convolutuion neural network in our model 
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform', activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform', activation='linear', border_mode='valid', bias=True))

    # use Adam alghorithm for optimization, with learning rate 0.0003 for all layers. #
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict(test_dir, IMG_NAME):
    srcnn_model = predict_model()
    srcnn_model.load_weights("model.h5")
    INPUT_NAME = "./Y_images/" + IMG_NAME
    OUTPUT_NAME = "./output/" + IMG_NAME
    IMG_NAME = test_dir + IMG_NAME

    #####################################################
    # read input image converted it to form of YCrCb,   #
    # first enlarged the image by 2 and applied bicubic #
    # interpolation conberted back to 3 channel image   #
    # then saved image in Y_image folder later we will  #
    # this image as input image                         #
    #####################################################
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] / 2 , shape[0] / 2 ), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)


    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.

    # predict our output image
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)


    #######################
    # psnr of calculation #
    #######################
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR) # read image of input given
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR) # read image which is bicubic interpolated image of input image
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR) # read image which is the output image of the given input image
    im2 = cv2.resize(im2,None,fx=0.5, fy=0.5) # resize interpolated image as input image
    im3 = cv2.resize(im3,None,fx=0.5, fy=0.5) # resize output image as input image
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0] # converted image from 3 channel to YCrCb form
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0] # converted image from 3 channel to YCrCb form
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0] # converted image from 3 channel to YCrCb form

    # calculated psnr of interpolated image and output image with respect to input image
    print IMG_NAME + " bicubic:"
    print cv2.PSNR(im1, im2)
    print IMG_NAME + " srcnn: "
    print cv2.PSNR(im2, im3)

if __name__ == "__main__":
    test_dir = "./test/"
    names = os.listdir(test_dir)
    names = sorted(names)
    nums = names.__len__()
    img_names = os.listdir
    for i in range(nums):
        name = names[i]
        # print (test_dir + name)
        predict(test_dir, name)
        
