'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Flatten, Input, Reshape, TimeDistributed, Bidirectional
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
#from keras.metrics import a


#def batch_size_verifier

def np_array_pair_generator(data,labels,start_at=0): #shape is something like 1, 11520, 11
    index=0
    while 1:
        while index in range(start_at,data.shape[1]):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x = data[:,index,:] #first dim = 0 doesn't work.
            y = labels[index,:] #yield shape = (4,)
            #print("data shape: {}, x type: {}, y type:{}".format(data.shape,type(x),type(y)))
            x = np.reshape(x,(1,x.shape[0],x.shape[1]))
            y = np.reshape(y, (1, y.shape[0]))
            #print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x.shape, y.shape))
            #print("x: {}, y: {}".format(x,y))
            #print("index: {}".format(index))
            print("x: {}, y: {}, index: {}".format(x,y,index))
            index =+ 1
            yield (x, y)

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = sg_utils.get_shortest_length()
epochs = 25

# number of elements ahead that are used to make the prediction
lahead = 1

#raw_path = "/home/ihsan/Documents/thesis_generator/results/devin/to_process/" #needs the absolute path, no tildes!
#processed_path = "/home/ihsan/Documents/thesis_generator/results/devin"

train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
seq_length_dict_filename = train_path + "/data/seq_length_dict.json"

#the testing part
print("TESTING PHASE")
model = load_model('try1_complete_model.h5')

# load data multiple times.
data_filenames = os.listdir(test_path + "data")
# print("before sorting, data_filenames: {}".format(data_filenames))
data_filenames.sort()
# print("after sorting, data_filenames: {}".format(data_filenames))


label_filenames = os.listdir(test_path + "label")
label_filenames.sort()
# print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames, label_filenames)
# print("before shuffling: {}".format(combined_filenames))
shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.

i=0
for files in combined_filenames:
    i=i+1
    data_load_path = test_path + '/data/' + files[0]
    label_load_path = test_path + '/label/' + files[1]
    # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
    test_array = np.load(data_load_path)
    label_array = np.load(label_load_path)[:, 1:]
    test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
    # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
    print("data/label shape: {}, {}".format(test_array.shape, label_array.shape))

    generator_starting_index = test_array.shape[
                                   1] - 1 - batch_size  # steps per epoch is how many times that generator is called
    #GENERATOR STARTING INDEX SHOULD REALLY BE RE EVALUATED!!
    test_generator = np_array_pair_generator(test_array, label_array, start_at = generator_starting_index)
    for i in range (batch_size):
        x = test_array[:, i, :]  # first dim = 0 doesn't work.
        y = label_array[i, :]  # yield shape = (4,)
        # print("data shape: {}, x type: {}, y type:{}".format(data.shape,type(x),type(y)))
        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        y = np.reshape(y, (1, y.shape[0]))
        X_test_batch, y_test_batch = x,y
        print("X_test_batch: {}, y_test_batch: {}".format(X_test_batch,y_test_batch))
        score = model.test_on_batch(X_test_batch,y_test_batch)
        print("Score: {}".format(score))
    #scores = model.evaluate_generator(test_generator, steps=10)
    #predictions = model.predict_generator(test_generator,steps=1)
    #print("scores: {}".format(scores))
    #np.savetxt("FirstResults_" + str(i) + ".npy", scores)
    #print("predictions: {}".format(predictions))
    #np.savetxt("FirstPredictions_" + str(i) + ".npy", predictions)
    #model.fit(train_array, label_array)
#print('Predicting')
#predicted_output = model.predict(cos, batch_size=batch_size)
'''print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(label_array)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()'''
