import re
import sys
import pdb
import csv
import random
import scipy.io
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler

def main(argv, argc):
    if argc != 1:
        print("Usage: python main.py <directory of files of cars>")
        exit(1)

    train_df, test_df = file_IO(argv)
    return 0

# ------------------------------------------------------------------------------
# read files
def file_IO(argv):
    print("Performing file I/O...\n\n")
    train_df = parse_annos_file("devkit/cars_train_annos.csv", True)
    test_df = parse_annos_file("devkit/cars_test_annos.csv", False)
    return train_df, test_df


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
                train_photo_info.append(loc)
            else:
                flag = 1
    if train:
        headers = ['min_x', 'max_x', 'min_y', 'max_y', 'class', 'file']
    else:
        headers = ['min_x', 'max_x', 'min_y', 'max_y', 'file']

    df = pd.DataFrame(mult_photos_info, columns = headers)
    return df

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))


# convert from .mat to .csv
#
# mat = scipy.io.loadmat('devkit/cars_train_annos.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# data.to_csv("devkit/cars_train_annos.csv")
