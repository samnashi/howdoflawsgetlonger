from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing


#def batch_size_verifier

#you limit the # of calls keras calls the generator OUTSIDE the generator.
#each time you fit, dataset length // batch size. round down!
def np_array_pair_generator(data,labels,start_at=0,generator_batch_size=64,scaled=True,scaler_type = 'standard',scale_what = 'data'): #shape is something like 1, 11520, 11
    '''Custom batch-yielding generator for Scattergro Output. You need to feed it the numpy array after running "Parse_Individual_Arrays script
    data and labels are self-explanatory.
    Parameters:
        start_at: configures where in the arrays do the generator start yielding (to ensure an LSTM doesn't always start at the same place
        generator_batch_size: how many "rows" of the numpy array does the generator yield each time
        scaled: whether the output is scaled or not.
        scaler_type: which sklearn scaler to call
        scale_what = either the data/label (the whole array), or the yield.'''
    if scaled == True:
        if scaler_type == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
            print('standard scaler initialized: {}'.format(scaler))
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        print("scaled: {}, scaler_type: {}".format(scaled,scaler_type))
        data_scaled = scaler.fit_transform(X=data,y=None)
        labels_scaled = scaler.fit_transform(X=labels,y=None)
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))
        data_scaled = np.expand_dims(data_scaled, axis=0) #add 1 dimension in the
        labels_scaled = np.expand_dims(labels_scaled,axis=0)
        index = start_at
    while 1: #for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        # for index in range(start_at,data_scaled.shape[1]):
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x = (data_scaled[:,index:index+generator_batch_size,:]) #first dim = 0 doesn't work.
        y = (labels_scaled[:,index:index+generator_batch_size,:]) #yield shape = (4,)
        index = index + generator_batch_size
        #print("data shape: {}, x type: {}, y type:{}".format(data_scaled.shape,type(x),type(y)))
        #x = np.reshape(x,(1,x.shape[0],x.shape[1]))
        #y = np.reshape(y, (1, y.shape[0],y.shape[1]))
        print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x.shape, y.shape))
        #print("x: {}, y: {}".format(x,y))
        #print("index: {}".format(index))
        if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): return
        #if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): raise StopIteration
        #assert(x.shape[1]==generator_batch_size)
        #assert(y.shape[1]==generator_batch_size)
        yield (x, y)

#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 1 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 1 #how many times the training corpus is sampled.
generator_batch_size = 4
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_run_2b_new_gen_short_"
Base_Path = "./"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#11 input columns
#4 output columns.

np.random.seed(1337)
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
#print("before shuffling: {}".format(combined_filenames))
shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.
#
# #define the model first
# a = Input(shape=(None,11))
# b = Bidirectional(LSTM(350,return_sequences=True))(a)
# c = Bidirectional(LSTM(350,return_sequences=True))(b)
# d = TimeDistributed(Dense(64, activation='selu'))(c) #timedistributed wrapper gives None,64
# e = TimeDistributed(Dense(32, activation='selu'))(d)
# out = TimeDistributed(Dense(4))(e)
#
# model = Model(inputs=a,outputs=out)
# print("Model summary: {}".format(model.summary()))
# model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy','mae','mape','mse'])
#
# print("Inputs: {}".format(model.input_shape))
# print ("Outputs: {}".format(model.output_shape))
# print ("Metrics: {}".format(model.metrics_names))
#
# plot_model(model, to_file='model_' + identifier + '.png',show_shapes=True)
# #print ("Actual input: {}".format(data.shape))
# #print ("Actual output: {}".format(target.shape))

print('loading data...')

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    print("TRAINING PHASE")

    for i in range(0,num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        #-----------------------------FOR TESTING ONLY------------------------
        train_array = train_array[0:12,:]
        label_array = label_array[0:12,:]
        #train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape,label_array.shape, i))
        train_generator = np_array_pair_generator(train_array,label_array,start_at=0,generator_batch_size=generator_batch_size)
        print("endlimit: {}".format(train_array.shape[0]//generator_batch_size))
        for i in range(0,(train_array.shape[0]//generator_batch_size)):
            print(i,train_generator.next()[0].shape)
        for i in range(0, (train_array.shape[0] // generator_batch_size)):
            print("second call")
            print(i, train_generator.next()[0].shape)
            # print(2, train_generator.next()[0].shape)
            # print(3, train_generator.next()[0].shape)
            # print(4, train_generator.next()[0].shape)
        # output1 = train_generator.next()
        # output2 = train_generator.next()
        # output3 = train_generator.next()
        # output4 = train_generator.next()
        # output5 = train_generator.next()
        # output6 = train_generator.next()
        # output7 = train_generator.next()
        # output8 = train_generator.next()
        # output9 = train_generator.next()
        # output10 = train_generator.next()
        # output11 = train_generator.next()
        # output12 = train_generator.next()
        # print("output 12 shape: {}".format(output12[0].shape))
        # #
        # output13 = train_generator.next()
        # print("output 13 shape: {}".format(output13[0].shape))
        # output14 = train_generator.next()
        # print(output5[0] == output1[0])
        # print(output2[0] == output1[0])