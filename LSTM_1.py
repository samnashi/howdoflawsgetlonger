'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
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


#def batch_size_verifier

def np_array_pair_generator(data,labels,start_at=0): #shape is something like 1, 11520, 11
    while 1:
        for index in range(start_at,data.shape[1]):
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
#11 input columns
#4 output columns.



#load data multiple times.
data_filenames = os.listdir(train_path + "data")
#print("before sorting, data_filenames: {}".format(data_filenames))
data_filenames.sort()
#print("after sorting, data_filenames: {}".format(data_filenames))


label_filenames = os.listdir(train_path + "label")
label_filenames.sort()
#print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames,label_filenames)
#print("before shuffling: {}".format(combined_filenames))
shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.

'''print('Creating Model...')
model = Sequential()
model.add(LSTM(64,
               input_shape = (None,11),
               return_sequences=True,
               stateful=False))
model.add(LSTM(64,return_sequences=False,stateful=False))
#model.add(Flatten())
#model.add(Dense(128))
model.add(Dense(4))
model.compile(loss='mse', optimizer='rmsprop')

model.summary()'''
#define the model first
#a = Input(batch_shape=(batch_size,None,11))
a = Input(shape=(None,11))
#b = LSTM(128,return_sequences=False)(a)
b = Bidirectional(LSTM(256,return_sequences=True))(a)
c = Bidirectional(LSTM(256,return_sequences=True))(b)
d = Bidirectional(LSTM(256,return_sequences=False))(c)
#c = Flatten()(c)
#c = Reshape(target_shape=(11520,5))(c)
d = TimeDistributed(Dense(64,activation='selu'))(b) #timedistributed wrapper gives None,64
#out = TimeDistributed(Dense(4))(c)
out = TimeDistributed(Dense(4))(d)

model = Model(inputs=a,outputs=out)
print("Model summary: {}".format(model.summary()))
model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy','mae','mape','mse'])

print("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Metrics: {}".format(model.metrics_names))

plot_model(model, to_file='model.png',show_shapes=True)
#print ("Actual input: {}".format(data.shape))
#print ("Actual output: {}".format(target.shape))

print('loading data...')

if os.path.isfile('./try_1.h5') == False:
    print("TRAINING PHASE")
    #tuples
    for files in combined_filenames:
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        print("data/label shape: {}, {}".format(train_array.shape,label_array.shape))

        generator_starting_index = train_array.shape[1] - 1 - batch_size #steps per epoch is how many times that generator is called
        model.fit_generator(np_array_pair_generator(train_array,label_array,start_at=generator_starting_index),epochs=50,steps_per_epoch=batch_size,verbose=2)

    model.save_weights('try_1.h5')
    model.save('try1_complete_model.h5')

if os.path.isfile('./try_1.h5') == True:
    #the testing part
    print("TESTING PHASE")
    model.load_weights('try_1.h5')

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
        # for i in range (batch_size):
        #     X_test_batch, y_test_batch = test_generator.next()
        #     score = model.predict_on_batch(X_test_batch,y_test_batch)
        #     print("Score: {}".format(score))
        scores = model.evaluate_generator(test_generator, steps=batch_size,max_queue_size=batch_size,use_multiprocessing=True)
        print("scores: {}".format(scores))
        #predictions = model.predict_generator(test_generator,steps=1)
        #print("predictions: {}".format(predictions))
        #np.savetxt("FirstPredictions_" + str(i) + ".npy", predictions)
        np.savetxt("FirstResults_" + str(i) + ".npy", scores)
        #print("predictions: {}".format(predictions))

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

'''hist=full_model.fit_generator(train_generator,samples_per_epoch=nb_training_samples,nb_epoch=1000,
                                      validation_data=validation_generator, nb_val_samples=42,
                                      callbacks=training_callbacks,
                                      class_weight=class_weight_to_fit)

        #class_weight=class_weight_to_fit

        best_epoch = np.argmin(np.asarray(hist.history['val_loss']))
        best_result = np.asarray((best_epoch,(np.asarray(hist.history['acc'])[best_epoch]),
                                 (np.asarray(hist.history['loss'])[best_epoch]),
                                (np.asarray(hist.history['val_acc'])[best_epoch]),
                                 (np.asarray(hist.history['val_loss'])[best_epoch])))
        print('best epoch index: {}, best result: {}'.format(best_epoch, best_result)) #actual epoch is index+1 because arrays start at 0..

        # # saves the best epoch's results
        np.savetxt(Base_Path + 'Results/BestEpochResult' + ppmethod + '.txt', best_result,
                   fmt='%5.6f', delimiter=' ', newline='\n', header='epoch, acc, loss, val_acc, val_loss',
                   footer=str(ppmethod), comments='# ')

        np.save(Base_Path + 'Results/acc' + ppmethod + '.npy', np.asarray(hist.history['acc']))
        np.save(Base_Path + 'Results/loss' + ppmethod + '.npy', np.asarray(hist.history['loss']))
        np.save(Base_Path + 'Results/val_acc' + ppmethod + '.npy', np.asarray(hist.history['val_acc']))
        np.save(Base_Path + 'Results/val_loss' + ppmethod + '.npy', np.asarray(hist.history['val_loss']))

        # # summarize history for accuracy
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model accuracy' + str(ppmethod))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Base_Path + 'Results/acc' + ppmethod + '.png', bbox_inches='tight')
        plt.clf()

        # # summarize history for loss
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss' + str(ppmethod))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Base_Path + 'Results/loss' + ppmethod + '.png', bbox_inches='tight')
        plt.clf()
'''