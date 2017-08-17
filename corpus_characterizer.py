from __future__ import print_function
import numpy as np
from random import shuffle
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing
from scattergro_parser_each import parse_scattergro

'''Characterizes sequences. Calculates the relevant statistics.'''

#call parser
parse_scattergro(analysis_mode = True, save_arrays = False, feature_identifier = 'fvx')




'''
#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 3 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 300 #how many times the training corpus is sampled.
generator_batch_size = 128
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_fv1a_"
Base_Path = "./"
#train_path = "/home/ihsan/Documents/thesis_models/train/"
#test_path = "/home/ihsan/Documents/thesis_models/test/"
train_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/train/'
test_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/'
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#11 input columns
#4 output columns.

np.random.seed(1337)
data_filenames = os.listdir(train_path + "data")
data_filenames.sort()

label_filenames = os.listdir(train_path + "label")
label_filenames.sort() #sorting makes sure the label and the data are lined up.
#print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames,label_filenames)
#print("before shuffling: {}".format(combined_filenames))
#shuffle(combined_filenames)
#print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.


#weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#HARDCODED

list_of_train_rows = []
row_dict = {}
weights_present_indicator = False
if weights_present_indicator == False:
    print("TRAINING PHASE")

    for item in combined_filenames:
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        #-----------COMMENT THESE OUT IF YOU WANT RESCALER ON----------------------------------
        #train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        #--------------------------------------------------------------------------------------
        print("filename: {}, data/label shape: {}, {}".format(str(files[0]),train_array.shape,label_array.shape))
        row_dict = {}
        row_dict['filename'] = str(files[0])
        row_dict['length'] = train_array.shape[1]
        row_dict['shape'] = str(train_array.shape)
        list_of_train_rows.append(row_dict)

    train_rows_df = pd.DataFrame(list_of_train_rows,columns = ['filename', 'length','shape'])
    train_rows_df.to_csv('train_characteristics' + identifier + '.csv')
#weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#-------------------------TESTING PHASE-------------------------------------------------------------
list_of_test_rows = []
row_dict = {}
weights_present_indicator = True
if weights_present_indicator == True:
    #data_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/data/')
    data_filenames = os.listdir(test_path + "data")
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    #label_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/label')
    label_filenames = os.listdir(test_path + "label")
    label_filenames.sort()
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_filenames))
    #shuffle(combined_filenames)

    i=0
    #TODO: still only saves single results.
    score_rows_list = []
    for files in combined_filenames:
        i=i+1
        # data_load_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/data/' + files[0]
        # label_load_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/label/' + files[1]
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        print("filename: {}, data/label shape: {}, {}".format(str(files[0]),test_array.shape, label_array.shape))
        predictions_length = generator_batch_size * (label_array.shape[0]//generator_batch_size)
        #largest integer multiple of the generator batch size that fits into the length of the sequence.
        row_dict = {}
        row_dict['filename'] = str(files[0])
        row_dict['length'] = test_array.shape[1]
        row_dict['shape'] = str(test_array.shape)
        score_rows_list.append(row_dict)

    test_rows_df = pd.DataFrame(score_rows_list, columns=score_rows_list[0].keys())
    test_rows_df.to_csv('test_characteristics_' + identifier + '.csv')

    #score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())


'''