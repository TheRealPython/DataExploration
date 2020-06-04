# basic libraries
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# for saving model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

# for model training
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras.backend as K
# Image Ordering changes the dimensions!
#K.set_image_dim_ordering('th')
keras.backend.image_data_format()
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

# tensorflow is integrated in keras
import tensorflow as tf

# for visualisation of loading bar
from tqdm import tqdm

# for time measure
import time

# basic functions for run_training()
from preprocessing import open_png_pictures, load_training_images, data_preparation
from plot_training_stats import plot_model_accuracy, plot_model_loss

def baseline_model(num_classes, loss="binary_crossentropy", optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy']):
    #Adam: optimizer = Adam(lr=0.0001, beta_1=0.9, epsilon=0.1), metrics=['accuracy'])
    # optimizer = Adam(lr=0.0001), metrics=['accuracy'])
    #SGD: optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    
    # create model architecture
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(3, 50, 50), activation='relu', padding="same"))
    #model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'), padding="same")
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'), padding="same")
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    
    # model.add(Conv2D(128, kernel_size=(3,3), activation='relu'), padding="same")
    # model.add(Conv2D(128, kernel_size=(3,3), activation='relu'), padding="same")
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics=metrics)
    # https://keras.io/optimizers/
    return model

def scores(model, X_test, y_test):
    # test data results
    scores = model.evaluate(X_test, y_test, verbose=0)
    return scores

def save_model(model):
    # serialize model structure to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    return None

def run_training(model_json_file="model.json", model_weights_file="model_weights.h5"):
    # mode train or test
    print('load')

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #check GPU support
    # from keras import backend as K
    # K.tensorflow_backend._get_available_gpus()
    print('gpuuuus')

    #print(K.tensorflow_backend._get_available_gpus())

    # fixing error message
    #keras.backend.get_session().run(tf.global_variables_initializer())
    # load components (training set, test set, num classes)
    num_classes, X_train, X_test, y_train, y_test = data_preparation()

    # build the model
    model = baseline_model(num_classes)

    # save weights
    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=callbacks_list, verbose = 1)

    # plots of the learning curves
    print(plot_model_accuracy(history))
    print(plot_model_loss(history))
        
    # save model
    save_model(model)
    
    keras.backend.clear_session()
    return scores(model, X_test, y_test)