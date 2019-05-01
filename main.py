import re
import sys
import pdb
import csv
import random
import scipy.io
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main(argv, argc):
    if argc != 1:
        print("Usage: python main.py <directory of files of cars>")
        exit(1)

    file_IO(argv)
    return 0

# ------------------------------------------------------------------------------
# read files
def file_IO(argv):
    print("Performing file I/O...\n\n")
    parse_train_annos_file("devkit/cars_train_annos.csv")
    mat = scipy.io.loadmat('devkit/cars_train_annos.mat')
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv("devkit/cars_train_annos.csv")

    #car_directory = (glob.glob(argv[1]+"/*.png"))
    #  TODO
    #1 read in photo files
    #2 find the features for classifying cars
    #  class: Make
    #return car_directory

# ------------------------------------------------------------------------------
# returns dataFrame of the training photos
def parse_train_annos_file(cars_train_annos):
    train_photo_info = []
    with open(cars_train_annos) as csvfile:
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
                train_photo_info.append(loc)
            else:
                flag = 1
    df = pd.DataFrame(train_photo_info, columns = ['min_x', 'max_x', 'min_y', 'max_y', 'class', 'file'])
    pdb.set_trace()
    return df

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
