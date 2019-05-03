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
import time
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

PHOTO_DIRECTORY = "cars_train/"
PHOTO_ANNOS = "devkit/cars_train_annos.csv"
OUT_FILENAME = "results.csv"
CAR_CLASSES_FILE = "devkit/cars_meta.csv"
SCALE_WIDTH = -1
SCALE_HEIGHT = -1
NUM_CLASSES = 0
EPOCH_SIZE = -1
CLASSES = []

def main(argv, argc):
    if argc != 4:
        print("Usage: python main.py <train-rate> <img-scaling> <epoch-size>")
        exit(1)

    io_time_begin = time.time()
    x_train, y_train, x_test, y_test = file_IO(argv)
    io_time_end = time.time()

    arch_time_begin = time.time()
    model = define_model_architecture(x_train, y_train, x_test, y_test)
    arch_time_end = time.time()

    fit_time_begin = time.time()
    model = compile_fit_model(model, x_train, y_train)
    fit_time_end = time.time()

    eval_time_begin = time.time()
    loss, accuracy = evaluate_model(model, x_test, y_test)
    eval_time_end = time.time()

    # io time in seconds
    io_time = io_time_end - io_time_begin

    # arch time in seconds
    arch_time = arch_time_end - arch_time_begin

    # fit time in seconds
    fit_time = fit_time_end - fit_time_begin

    # evaluation time in seconds
    evaluation_time = eval_time_end - eval_time_begin

    strToWrite = str(EPOCH_SIZE)        + ", " + \
                 str(SCALE_WIDTH)       + ", " + \
                 str(SCALE_HEIGHT)      + ", " + \
                 argv[1]                + ", " + \
                 str(loss)              + ", " + \
                 str(accuracy)          + ", " + \
                 str(io_time)           + ", " + \
                 str(arch_time)         + ", " + \
                 str(fit_time)          + ", " + \
                 str(evaluation_time)   + "\n"

    file = open(OUT_FILENAME, "a")
    file.write(strToWrite)
    file.close()

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

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

# ------------------------------------------------------------------------------
# Compile and fit the model
# nb_epoch ~ number of times gone through the training set
def compile_fit_model(model, x_train, y_train):
    global EPOCH_SIZE
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, nb_epoch=EPOCH_SIZE, verbose=1)

    return model

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


# ------------------------------------------------------------------------------
# read files
def file_IO(argv):
    global PHOTO_ANNOS, SCALE_WIDTH, SCALE_HEIGHT, EPOCH_SIZE
    find_classes()
    train_rate = float(argv[1])
    SCALE_WIDTH = int(argv[2])
    SCALE_HEIGHT = int(argv[2])
    EPOCH_SIZE = int(argv[3])
    print("Performing file I/O...\n\n")
    df = parse_annos_file(PHOTO_ANNOS, True)
    df["class"] = update_y(df["class"])
    x_train, y_train, x_test, y_test = preprocess_data(df, train_rate)

    # need to read the img files
    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------------------------
# Find the indices of unique car brands
def find_classes():
    global CAR_CLASSES_FILE, CLASSES, NUM_CLASSES

    with open(CAR_CLASSES_FILE) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        temp = ""
        itr, flag = 0, 0
        for row in readCSV:
            if flag == 1:
                cur_brand = row[1].split()[0]
                if cur_brand != temp:
                    CLASSES.append(itr)
                    temp = cur_brand
                itr += 1
            else:
                flag = 1
    NUM_CLASSES = len(CLASSES)

def update_y(class_column):
    global CLASSES
    class_itr = 0
    for i in range(class_column.shape[0]):
        for j in range(len(CLASSES)):
            if int(class_column[i]) <= CLASSES[j]:
                class_column[i] = str(j - 1)
                break
    return class_column

# ------------------------------------------------------------------------------
# rescale cropped images
def rescale(img, width, height):
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------------------------
# preprocess input and output data
def preprocess_data(df, train_rate):
    global SCALE_WIDTH, SCALE_HEIGHT, NUM_CLASSES, PHOTO_DIRECTORY
    features = []
    #    df = df.sample(frac=1)
    for index, row in df.iterrows():
        min_x, max_x = int(row["min_x"]), int(row['max_x'])
        min_y, max_y = int(row['min_y']), int(row['max_y'])

        img = cv2.imread(PHOTO_DIRECTORY + row['file'])

        crop_img = img[min_y:max_y, min_x:max_x]
        crop_img = rescale(crop_img, SCALE_WIDTH, SCALE_HEIGHT)
        features.append(crop_img)

    features = np.asarray(features)
    num_train = int(df.shape[0]*train_rate)
    x_train, y_train = features[:num_train], np.asarray(df[:num_train][df.columns[-2]]).astype("int")
    x_test, y_test = features[num_train:], np.asarray(df[num_train:][df.columns[-2]]).astype("int")

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    y_train = np_utils.to_categorical(y_train,NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test,NUM_CLASSES)

    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------------------------
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
