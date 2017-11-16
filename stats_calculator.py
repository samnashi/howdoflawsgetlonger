from scipy.stats import describe, kurtosistest, skewtest, normaltest
from corpus_characterizer import generator_chunker
import numpy as np
import os

#@@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"

#---------part from covariance tester---------------

CHUNKER_BATCH_SIZE = 128
#load data multiple times.
data_filenames = os.listdir(train_path + "data")
#print("before sorting, data_filenames: {}".format(data_filenames))
data_filenames.sort()
#print("after sorting, data_filenames: {}".format(data_filenames))

label_filenames = os.listdir(train_path + "label")
label_filenames.sort() #sorting makes sure the label and the data are lined up.
#print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames,label_filenames)

#for index_to_load in range(0,len(combined_filenames)):
for index_to_load in range(0,2):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    if train_array.shape[1] > 11:
        train_array = train_array[:,1:]
    if label_train_array.shape[1] > 5:
        label_train_array = label_train_array[:,1:]
    identifier = files[0][:-4]
    for col in range (0,train_array.shape[1]):
        print("train array col {}: {}".format(col,describe(train_array[:,col],axis=0)))
        print("kurtosis {}".format(kurtosistest(train_array[:, col], axis=0)))
        print("skew {}".format(skewtest(train_array[:, col], axis=0)))
        print("normal {}".format(normaltest(train_array[:, col], axis=0)))
    for col in range (0,label_train_array.shape[1]):
        print("train label array col {}: {}".format(col,describe(label_train_array[:,col],axis=0)))
        print("kurtosis {}".format(kurtosistest(label_train_array[:, col], axis=0)))
        print("skew {}".format(skewtest(label_train_array[:, col], axis=0)))
        print("normal {}".format(normaltest(label_train_array[:, col], axis=0)))

        #TODO: cast each column into a dataframe row.

    