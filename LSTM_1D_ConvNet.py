'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Conv1D
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils


# def batch_size_verifier

def np_array_pair_generator(data, labels, start_at=0):  # shape is something like 1, 11520, 11
    while 1:
        for index in range(start_at, data.shape[1]):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x = (data[:, index, :])  # first dim = 0 doesn't work.
            y = (labels[:, index, :])  # yield shape = (4,)
            # print("data shape: {}, x type: {}, y type:{}".format(data.shape,type(x),type(y)))
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
            y = np.reshape(y, (1, y.shape[0], y.shape[1]))
            # print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x.shape, y.shape))
            # print("x: {}, y: {}".format(x,y))
            # print("index: {}".format(index))
            yield (x, y)


# !!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
batch_size = sg_utils.get_shortest_length()  # a suggestion. will also print the remainders.
num_epochs = 3  # individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 200  # how many times the training corpus is sampled.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_256bd_16d_tanh_adagrad"
Base_Path = "./"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
# 11 input columns
# 4 output columns.

# load data multiple times.
data_filenames = os.listdir(train_path + "data")
# print("before sorting, data_filenames: {}".format(data_filenames))
data_filenames.sort()
# print("after sorting, data_filenames: {}".format(data_filenames))

label_filenames = os.listdir(train_path + "label")
label_filenames.sort()
# print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames, label_filenames)
# print("before shuffling: {}".format(combined_filenames))
shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.

# define the model first
a = Input(shape=(None, 11))
b1 = Conv1D()
b = Bidirectional(LSTM(128, return_sequences=True))(a)
c = Bidirectional(LSTM(128, return_sequences=True))(b)
d = TimeDistributed(Dense(16, activation='tanh'))(c)  # timedistributed wrapper gives None,64
out = TimeDistributed(Dense(4))(d)

model = Model(inputs=a, outputs=out)
print("Model summary: {}".format(model.summary()))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mae', 'mape', 'mse'])

print("Inputs: {}".format(model.input_shape))
print("Outputs: {}".format(model.output_shape))
print("Metrics: {}".format(model.metrics_names))

plot_model(model, to_file='model_' + identifier + '.png', show_shapes=True)
# print ("Actual input: {}".format(data.shape))
# print ("Actual output: {}".format(target.shape))

print('loading data...')

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    print("TRAINING PHASE")

    # tuples

    for i in range(0, num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        train_array = np.reshape(train_array, (1, train_array.shape[0], train_array.shape[1]))
        label_array = np.reshape(label_array,
                                 (1, label_array.shape[0], label_array.shape[1]))  # label needs to be 3D for TD!
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape, label_array.shape, i))

        generator_starting_index = train_array.shape[
                                       1] - 1 - batch_size  # steps per epoch is how many times that generator is called
        training_hist = model.fit_generator(np_array_pair_generator(train_array, label_array, start_at=0),
                                            epochs=num_epochs, steps_per_epoch=train_array.shape[1], verbose=2)

    # model.save('Model_' + str(num_sequence_draws) + identifier + '.h5')
    model.save_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')
    print('training_hist keys: {}'.format(training_hist.history.keys()))

    best_epoch = np.argmax(np.asarray(training_hist.history['acc']))

    best_result = np.asarray((best_epoch, (np.asarray(training_hist.history['loss'])[best_epoch]),
                              (np.asarray(training_hist.history['acc'])[best_epoch]),
                              (np.asarray(training_hist.history['mean_absolute_percentage_error'])[best_epoch]),
                              (np.asarray(training_hist.history['mean_absolute_error'])[best_epoch])))
    print('best epoch index: {}, best result: {}'.format(best_epoch,
                                                         best_result))  # actual epoch is index+1 because arrays start at 0..

    # # saves the best epoch's results
    np.savetxt(Base_Path + 'results/BestEpochResult_' + str(num_sequence_draws) + identifier + '.txt', best_result,
               fmt='%5.6f', delimiter=' ', newline='\n', header='epoch, loss, acc, mape, mae',
               footer=str(), comments='# ')

    np.save(Base_Path + 'results/acc_' + str(num_sequence_draws) + identifier + '.npy',
            np.asarray(training_hist.history['acc']))
    np.save(Base_Path + 'results/loss_' + str(num_sequence_draws) + identifier + '.npy',
            np.asarray(training_hist.history['loss']))

    # # summarize history for accuracy
    plt.plot(training_hist.history['acc'])
    plt.title('model MAIN accuracy' + identifier)
    plt.ylabel('MAIN accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(Base_Path + 'results/main_acc_' + str(num_sequence_draws) + identifier + '.png', bbox_inches='tight')
    plt.clf()

    # # summarize history for loss
    plt.plot(training_hist.history['loss'])
    plt.title('model loss' + identifier)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(Base_Path + 'results/loss_' + str(num_sequence_draws) + identifier + '.png', bbox_inches='tight')
    plt.clf()

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == True:
    # the testing part
    print("TESTING PHASE, with weights {}".format('Weights_' + str(num_sequence_draws) + identifier + '.h5'))
    model.load_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')

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

    i = 0
    # TODO: still only saves single results.
    for files in combined_filenames:
        i = i + 1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        label_array = np.reshape(label_array,
                                 (1, label_array.shape[0], label_array.shape[1]))  # label doesn't need to be 3D
        print("data/label shape: {}, {}".format(test_array.shape, label_array.shape))

        generator_starting_index = test_array.shape[
                                       1] - 1 - batch_size  # steps per epoch is how many times that generator is called
        # GENERATOR STARTING INDEX SHOULD REALLY BE RE EVALUATED!!
        test_generator = np_array_pair_generator(test_array, label_array, start_at=0)
        # for i in range (batch_size):
        #     X_test_batch, y_test_batch = test_generator.next()
        #     score = model.predict_on_batch(X_test_batch,y_test_batch)
        #     print("Score: {}".format(score))
        score = model.evaluate_generator(test_generator, steps=test_array.shape[1], max_queue_size=test_array.shape[1],
                                         use_multiprocessing=True)
        print("scores: {}".format(score))
        np.savetxt(Base_Path + 'results/TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
                   fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
                   footer=str(), comments='# ')


