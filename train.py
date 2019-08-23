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

#############################################
# model object which is model of our method #
#############################################
def model():
    SRCNN = Sequential() 

    # adding 3 convolutuion neural network in our model 
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform', activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform', activation='linear', border_mode='valid', bias=True))
    
    # use Adam algorithm for optimization, with learning rate 0.0003 for all layers. #
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

def train():
    srcnn_model = model()
    print(srcnn_model.summary())

    ###################################################################################################
    # get training and testing data with there labels from .h5 file created by prepare_data.py module #
    ###################################################################################################
    data, label = pd.read_training_data("./train.h5")
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint("check.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    # fit our train and test data in model to train machine #
    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label), callbacks=callbacks_list, shuffle=True, nb_epoch=100, verbose=0)
    
    # to save our trained model we first save model in json file and then store it in .h5 file
    model_json = srcnn_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    srcnn_model.save_weights("model.h5")



if __name__ == "__main__":
    train()
