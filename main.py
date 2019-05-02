import re
import cv2
import sys
import pdb
import csv
import random
import scipy.io
import numpy as np
import pandas as pd
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

SCALE_WIDTH = 400
SCALE_HEIGHT = 400
NUM_CLASSES = 196

def main(argv, argc):
    if argc != 2:
        print("Usage: python main.py <train rate>")
        exit(1)

    x_train, y_train, x_test, y_test = file_IO(argv)
    model = define_model_architecture(x_train, y_train, x_test, y_test)
    model = compile_fit_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    # Reshape input data
    print(x_train.shape)
    pdb.set_trace()

    return 0

# ------------------------------------------------------------------------------
# Define model architecture
def define_model_architecture(x_train, y_train, x_test, y_test):
    global NUM_CLASSES
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape = (SCALE_WIDTH, SCALE_HEIGHT, 3)))
    model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    pdb.set_trace()

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    pdb.set_trace()
    return model

# ------------------------------------------------------------------------------
# Compile and fit the model
# nb_epoch ~ number of times gone through the training set
def compile_fit_model(model, x_train, y_train):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, nb_epoch=100, verbose=1)
    pdb.set_trace()

    # this is our score after running model.fit
    #hm, forgot what that means | me too... LOOL
    # i'll try running with the sample code given, had to fix a few errors, runnning now
    # the accuracy is 10%. Running with nb_epoch = 100  | ah ok, oh i see it says on the side when it's calculating
    # Here's what i got for the sample one from the site: [0.02759441680457744, 0.9923]
    # it seems like the right number is the
    return model

def evaluate_model(model, x_test, y_test):
    score, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("score: " + str(score))
    print("accuracy: " + str(accuracy))

# ------------------------------------------------------------------------------
# read files
def file_IO(argv):
    train_rate = float(argv[1])
    print("Performing file I/O...\n\n")
    df = parse_annos_file("devkit/mock_cars_train_annos.csv", True)
    x_train, y_train, x_test, y_test = preprocess_data(df, train_rate)

    # need to read the img files
    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------------------------
# rescale cropped images
def rescale(img, width, height):
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------------------------
# preprocess input and output data
def preprocess_data(df, train_rate):
    global SCALE_WIDTH, SCALE_HEIGHT, NUM_CLASSES
    features = []
    #    df = df.sample(frac=1)
    for index, row in df.iterrows():
        min_x, max_x = int(row["min_x"]), int(row['max_x'])
        min_y, max_y = int(row['min_y']), int(row['max_y'])

        img = cv2.imread("mock_cars_train/" + row['file'])

        crop_img = img[min_y:max_y, min_x:max_x]
        crop_img = rescale(crop_img, SCALE_WIDTH, SCALE_HEIGHT)
        features.append(crop_img)

        # crop_img = rescale(crop_img, SCALE_WIDTH, SCALE_HEIGHT)
        # cv2.imshow("scaled", crop_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    features = np.asarray(features)
    num_train = int(df.shape[0]*train_rate)
    x_train, y_train = features[:num_train], np.asarray(df[:num_train][df.columns[-2]]).astype("int")
    x_test, y_test = features[num_train:], np.asarray(df[num_train:][df.columns[-2]]).astype("int")

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # x_train = x_train.reshape(x_train.shape[0], SCALE_WIDTH, SCALE_HEIGHT, 3)
    # x_test = x_test.reshape(x_test.shape[0], SCALE_WIDTH, SCALE_HEIGHT, 3)
    y_train = np_utils.to_categorical(y_train,NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test,NUM_CLASSES)

    return x_train, y_train, x_test, y_test

# ---------------------------------------------------------i'm just getting set up since i just woke up haha---------------------
# returns dataFrame of (train/test) data
def parse_annos_file(cars_annos, train):
    mult_photos_info = []
    with open(cars_annos) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        flag = 0
        for row in readCSV:
            if flag == 1:
                loc = re.findall('\[.*?\]', row[1])
                for i in range(len(loc)):
                    if loc[i] == loc[-1]:
                        loc[i] = loc[i].strip('[\'')
                        loc[i] = loc[i].strip('\']')
                    else:
                        loc[i] = loc[i].strip('[[')
                        loc[i] = loc[i].strip(']')
                mult_photos_info.append(loc)
            else:
                flag = 1
    if train:
        headers = ['min_x', 'min_y', 'max_x', 'max_y', 'class', 'file']
    else:
        headers = ['min_x', 'max_y', 'max_x', 'min_y', 'file']

    df = pd.DataFrame(mult_photos_info, columns = headers)
    return df

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))

# ------------------------------------------------------------------------------
# Pseudocode
    # # Convert data type and normalize values
    # x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    #
    # x_train /= 255
    # x_test /= 255
    #
    # # Preprocess class labels
    # y_train, y_test = np_utils.to_categorical(y_train, 10), np_utils.to_categorical(y_test, 10)
    #
    # # Declare Sequential model
    # model = Sequential()
    #
    # # CNN input layer
    # model.add(Convolution2D(32, 3, 3), activation='relu', input_shape=(1, 28, 28))
    #
    # model.add(Convolution2D(32, 3, 3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))
    #
    # # Fully connected Dense layers
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    #
    # # Compile model
    # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #
    # # Fit model on training data
    # model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
    #
    # score = model.evaluate(x_test, y_test, verbose=0)
# ------------------------------------------------------------------------------

# convert from .mat to .csv
#
# mat = scipy.io.loadmat('devkit/cars_train_annos.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# data.to_csv("devkit/cars_train_annos.csv")
