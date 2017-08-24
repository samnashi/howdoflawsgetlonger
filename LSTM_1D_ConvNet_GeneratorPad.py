from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Dense, Dropout, \
    Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, concatenate, BatchNormalization
from keras.initializers import lecun_normal,glorot_normal
from keras import metrics
import pandas as pd
import scipy.io as sio
from keras.callbacks import CSVLogger
import os
import csv
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing
from LSTM_1D_ConvNet_Base import pair_generator_1dconv_lstm, conv_block_normal_param_count, conv_block_double_param_count, \
    conv_block_normal_param_count_narrow_window, conv_block_double_param_count_narrow_window,\
    conv_block_double_param_count_narrow_window_causal, conv_block_normal_param_count_narrow_window_causal, reference_bilstm

# np.set_printoptions(threshold='nan')


#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
num_epochs = 5 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 20 #how many times the training corpus is sampled.
generator_batch_size = 128
generator_batch_size_valid_x1 = (generator_batch_size+28)#4layer conv
generator_batch_size_valid_x2 = (generator_batch_size+14)
generator_batch_size_valid_x3 = (generator_batch_size+28)#4layer conv
generator_batch_size_valid_x4 = (generator_batch_size+14)
generator_batch_size_valid_x5 = (generator_batch_size+28)#4layer conv
generator_batch_size_valid_x6 = (generator_batch_size+14)
generator_batch_size_valid_x7 = (generator_batch_size+28)#4layer conv
generator_batch_size_valid_x8 = (generator_batch_size+14)
generator_batch_size_valid_x9 = (generator_batch_size+14)
generator_batch_size_valid_x10 = (generator_batch_size+14)
generator_batch_size_valid_x11 = (generator_batch_size+14)
finetune = False
use_precomp_sscaler = True
sequence_circumnavigation_amt = 3
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# identifier = "_convlstm_run1_" + str(generator_batch_size) + "b_completev1data_valid_4layer_1357_"
identifier = "_conv1d_run2_" + str(generator_batch_size) + "gen_pad"
#^^^^^^^^^^^^^^TO RUN ON CHEZ CHAN^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Base_Path = "/home/devin/Documents/PITTA LID/"
# image_path = "/home/devin/Documents/PITTA LID/img/"
# train_path = "/home/devin/Documents/PITTA LID/Train FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/Test FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/FV1b 1d nonlinear/"
#+++++++++++++++TO RUN ON LOCAL (IHSAN)+++++++++++++++++++++++++++++++
Base_Path = "/home/ihsan/Documents/thesis_models/"
image_path = "/home/ihsan/Documents/thesis_models/images"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#4 output columns.

np.random.seed(1337)

a1 = Input(shape=(None, 1))
a2 = Input(shape=(None, 1))
a3 = Input(shape=(None, 1))
a4 = Input(shape=(None, 1))
a5 = Input(shape=(None, 1))
a6 = Input(shape=(None, 1))
a7 = Input(shape=(None, 1))
a8 = Input(shape=(None, 1))
a9 = Input(shape=(None, 1))
a10 = Input(shape=(None, 1))
a11 = Input(shape=(None, 1))

# g1 = conv_block_double_param_count_narrow_window(input_tensor=a1)
# f2 = conv_block_normal_param_count_narrow_window(input_tensor=a2)
# g3 = conv_block_double_param_count_narrow_window(input_tensor=a3)
# f4 = conv_block_normal_param_count_narrow_window(input_tensor=a4)
# g5 = conv_block_double_param_count_narrow_window(input_tensor=a5)
# f6 = conv_block_normal_param_count_narrow_window(input_tensor=a6)
# g7 = conv_block_double_param_count_narrow_window(input_tensor=a7)
# f8 = conv_block_normal_param_count_narrow_window(input_tensor=a8)
# f9 = conv_block_normal_param_count_narrow_window(input_tensor=a9)
# f10 = conv_block_normal_param_count_narrow_window(input_tensor=a10)
# f11 = conv_block_normal_param_count_narrow_window(input_tensor=a11)

g1 = conv_block_double_param_count_narrow_window_causal(input_tensor=a1)
f2 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a2)
g3 = conv_block_double_param_count_narrow_window_causal(input_tensor=a3)
f4 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a4)
g5 = conv_block_double_param_count_narrow_window_causal(input_tensor=a5)
f6 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a6)
g7 = conv_block_double_param_count_narrow_window_causal(input_tensor=a7)
f8 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a8)
f9 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a9)
f10 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a10)
f11 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a11)

# define the model first

tensors_to_concat = [g1, f2, g3, f4, g5, f6, g7, f8, f9, f10, f11]
g = concatenate(tensors_to_concat)
out = reference_bilstm(input_tensor=g)


model = Model(inputs=[a1,a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], outputs=out)
plot_model(model, to_file='model_' + identifier + '.png',show_shapes=True)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mae', 'mape', 'mse'])
print("Model summary: {}".format(model.summary()))

print("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Metrics: {}".format(model.metrics_names))



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
print('loading data...')

print("weights present? {}".format((os.path.isfile(Base_Path + 'Weights_' +
                                                   str(num_sequence_draws) + identifier + '.h5'))))
if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    print("TRAINING PHASE")

    for i in range(0,num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        print("files: {}".format(files))
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape,label_array.shape, i))
        # train_array = np.reshape(train_array,(1,generator_batch_size,train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        train_generator = pair_generator_1dconv_lstm(train_array, label_array, start_at=0,
                                                     generator_batch_size=generator_batch_size,
                                                     use_precomputed_coeffs=use_precomp_sscaler)
        training_hist = model.fit_generator(train_generator, epochs=num_epochs,
                                            steps_per_epoch=sequence_circumnavigation_amt*
                                                            (train_array.shape[0]//generator_batch_size), verbose=2)

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier + '.h5'
    print("after {} iterations, model weights is saved as {}".format(num_sequence_draws*num_epochs, weights_file_name))
    model.save_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == True:
    #the testing part
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

    i=0
    scaler_output = sklearn.preprocessing.StandardScaler()
    #TODO: still only saves single results.
    for files in combined_filenames:
        csv_logger = CSVLogger('logtest.log')
        i=i+1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
        print(files[0])
        # print("Metrics: {}".format(model.metrics_names))
        # steps per epoch is how many times that generator is called
        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at = 0,
                                                    generator_batch_size=generator_batch_size,
                                                    use_precomputed_coeffs=use_precomp_sscaler)
        for i in range (1):
            X_test_batch, y_test_batch = test_generator.next()
            # print(X_test_batch)
            # print(y_test_batch)
            score = model.predict_on_batch(X_test_batch)
            # print("Score: {}".format(score)) #test_array.shape[1]//generator_batch_size
        score = model.evaluate_generator(test_generator, steps=(test_array.shape[0]//generator_batch_size),
                                         max_queue_size=test_array.shape[0],use_multiprocessing=False)
        print("scores: {}".format(score))
        # print(score)
        #home/ihsan/Documents/thesis_models/results/
        np.savetxt('TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
                   fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
                   footer=str(), comments='# ')

        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at = 0,
                                                    generator_batch_size=generator_batch_size,
                                                    use_precomputed_coeffs=use_precomp_sscaler)
        prediction_length = (int(0.85*(generator_batch_size * (label_array.shape[0]//generator_batch_size))))
        test_i=0
        # Kindly declare the shape
        x_prediction = np.zeros(shape=[1, prediction_length, 4])
        y_prediction = np.zeros(shape=[1, prediction_length, 4])


        while test_i <= prediction_length - generator_batch_size:
            x_test_batch, y_test_batch = test_generator.next()
            x_prediction[0,test_i:test_i + generator_batch_size,:] = model.predict_on_batch(x_test_batch)
            y_prediction[0,test_i:test_i + generator_batch_size,:] = y_test_batch
            test_i += generator_batch_size
        # print("array shape {}".format(y_prediction[0,int(0.95*prediction_length), :].shape))
        np.save(Base_Path + 'predictionbatch' + str(files[0]), arr=y_prediction)

        # print(y_prediction.shape)
        # print (x_prediction.shape)
        # print ("label array shape: {}".format(label_array.shape))

        # print("y_prediction shape: {}".format(y_prediction.shape))
        y_prediction_temp = y_prediction
        y_prediction = np.reshape(y_prediction,newshape=(y_prediction_temp.shape[1],y_prediction_temp.shape[2]))
        label_truth = label_array[0:y_prediction.shape[0],:]
        # print (label_truth.shape)
        label_truth_temp = label_truth
        label_truth = scaler_output.fit_transform(X=label_truth_temp,y=None)

        resample_interval = 16
        label_truth = label_truth[::resample_interval,:]
        y_prediction = y_prediction[::resample_interval,:]

        # print(len(y_prediction))


        # if (str(files[0]) == 'sequence_2c_288_9_fv1b.npy') == True:
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        #     plt.plot(label_truth[:, 0], '^', label="ground truth", markersize=5)
        #     plt.plot(y_prediction[:, 0], '.', label="prediction", markersize=4)
        #     plt.xscale('log')
        #     plt.xlabel('# Cycle(s)')
        #     plt.yscale('log')
        #     plt.ylabel('Value(s)')
        #     plt.legend()
        #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
        #     plt.title('truth vs prediction from 75% - 100% of the sequence on Crack 01')
        #     plt.grid(True)
        #     plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_75_100_newmarker_batch' + str(
        #         generator_batch_size) + '_.png')
        #
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        #     plt.plot(label_truth[:, 1], '^', label="ground truth", markersize=5)
        #     plt.plot(y_prediction[:, 1], 'v', label="prediction", markersize=4)
        #     plt.xscale('log')
        #     plt.xlabel('# Cycle(s)')
        #     plt.yscale('log')
        #     plt.ylabel('Value(s)')
        #     plt.legend()
        #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
        #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 02')
        #     plt.grid(True)
        #     plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_75_100_newmarker_batch' + str(
        #         generator_batch_size) + '_.png')
        #
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        #     plt.plot(label_truth[:, 2], '^', label="ground truth", markersize=5)
        #     plt.plot(y_prediction[:, 2], 'v', label="prediction", markersize=4)
        #     plt.xscale('log')
        #     plt.xlabel('# Cycle(s)')
        #     plt.yscale('log')
        #     plt.ylabel('Value(s)')
        #     plt.legend()
        #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
        #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 03')
        #     plt.grid(True)
        #     plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_75_100_newmarker_batch' + str(
        #         generator_batch_size) + '_.png')
        #
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        #     plt.plot(label_truth[:, 3], '^', label="ground truth", markersize=5)
        #     plt.plot(y_prediction[:, 3], 'v', label="prediction", markersize=4)
        #     plt.xscale('log')
        #     plt.xlabel('# Cycle(s)')
        #     plt.yscale('log')
        #     plt.ylabel('Value(s)')
        #     plt.legend()
        #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
        #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 04')
        #     plt.grid(True)
        #     plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_75_100_newmarker_batch' + str(
        #         generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,0],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,0],'.',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction from 50% - 100% of the sequence on Crack 01')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,1],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,1],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 02')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,2],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,2],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 03')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,3],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,3],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 04')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')