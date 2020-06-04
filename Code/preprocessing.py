# basic libraries
import numpy as np
import pandas as pd
import os
import numpy

# for data set and confusion matrix
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
# for image manipulation 
from PIL import Image

# for visualisation of loading bar
from tqdm import tqdm

def open_png_pictures(filename:str):
    # Methode ergänzen, die die Größe auslesen kann
    with Image.open(filename) as image:
        width, height = image.size
        if width != 50 or height != 50:
            return None
        else:
            return numpy.array(image.getdata()).reshape(3, 50, 50).astype(np.uint8)

def load_training_images(path:str):
    # num = num of classes
    # append all images and labels to seperated lists
    # path = ".\\breast-histopathology-images\IDC_regular_ps50_idx5"
    X_total = list()
    y_total = list()

    for patient in os.listdir(path):
        bpath = os.path.join(path, str(patient))
        for label in os.listdir(bpath):
            npath = os.path.join(bpath, str(label))
            for img in tqdm(os.listdir(npath)):
                #print(img)
                #print(os.path.join(npath,img))
                if open_png_pictures(os.path.join(npath,img)) is not None:
                    X_total.append(open_png_pictures(os.path.join(npath,img)))

                    y_total.append(label)
        
    print(len(X_total))
    print(len(y_total))

    # convert the image list to numpy array
    X_total = np.array(X_total)

    # normalize the arrays to 0 and 1
    X_total = X_total / 255 

    # to_categorical converts an array of labeled data(from 0 to nb_classes-1) to one-hot vector.
    y_total = np_utils.to_categorical(y_total)

    np.save('Xdata.npy', X_total) # save
    np.save('Ydata.npy', y_total) # save
    return None

def data_preparation():
    X_total = np.load('../Xdata.npy', allow_pickle=True) # load images
    y_total = np.load('../Ydata.npy') # load labels
    
    # num_classes for Dense Layer
    num_classes = y_total.shape[1]
    print('num classes:',num_classes)
    
    print("Test example array of the first image: ", X_total[0])
    
    print(f"length X_total: {len(X_total)}")
    print("---------------------------------------------------------------------------------------")
    print(f"length y_total: {len(y_total)}")
    print("---------------------------------------------------------------------------------------")
    
    # randomize the data set
    X_total, y_total = shuffle(X_total, y_total, random_state=0)

    # splitting the data set into training set (75%) and test set (25%)
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total)
    print("Dimensionen Traindata: ", X_test.ndim)
    
    return (num_classes, X_train, X_test, y_train, y_test)


#load_training_images(".\\breast-histopathology-images\IDC_regular_ps50_idx5")