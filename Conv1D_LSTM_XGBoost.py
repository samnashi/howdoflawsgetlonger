from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Dense, Dropout, \
    Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, concatenate, BatchNormalization
from keras.initializers import lecun_normal, glorot_normal
from keras.regularizers import l1, l1_l2, l2
from keras import metrics
from keras.optimizers import adam, rmsprop
import pandas as pd
import scipy.io as sio
from keras.callbacks import CSVLogger, TerminateOnNaN
import os
import csv
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing
import xgboost as xgb

from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"

def create_training_set():
    '''loads the input features and the labels'''
    # load data multiple times.
    data_filenames = os.listdir(train_path + "data")
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    label_filenames = os.listdir(train_path + "label")
    label_filenames.sort()  # sorting makes sure the label and the data are lined up.
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_filenames))
    shuffle(combined_filenames)
    print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.
    print('Data loaded.')

    return combined_filenames 

def create_testing_set():
    # load data multiple times.
    data_filenames = list(set(os.listdir(test_path + "data")))
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    label_filenames = list(set(os.listdir(test_path + "label")))
    label_filenames.sort()
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_test_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_test_filenames))
    shuffle(combined_test_filenames)
    print("after shuffling: {}".format(combined_test_filenames))  # shuffling works ok.

    return combined_test_filenames

if __name__ == "__main__":

    num_sequence_draws = 440
    GENERATOR_BATCH_SIZE = 128
    num_epochs = 1 #because you gotta feed the base model the same way you fed it during training... (RNN commandments)

    #create the data-pair filenames (using zip), use the helper methods
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    print(train_set_filenames)
    print(test_set_filenames)

    # load model
    identifier_post_training = "xgb_testmodel_relu_ca_tanh_da_3_cbd_standard_per_batch_sclr_l1l2_kr_HLR.h5"
    # './' + identifier_post_training + '.h5'
    raw_base_model = load_model("./model_" + identifier_post_training)
    print("model loaded.")

    for i in range(0, num_sequence_draws):
        index_to_load = np.random.randint(0, len(train_set_filenames))  # switch to iterations
        files = train_set_filenames[index_to_load]
        print("files: {}".format(files))
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        if train_array.shape[1] != 11:
            train_array = train_array[:, 1:]
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape, label_array.shape, i))

    # train_generator = pair_generator_1dconv_lstm_bagged(train_array, label_array, start_at=active_starting_position,
    #                                                     generator_batch_size=generator_batch_size,
    #                                                     use_precomputed_coeffs=use_precomp_sscaler,
    #                                                     scaled=scaler_active,
    #                                                     scaler_type=active_scaler_type)

        xgb_train_generator = pair_generator_1dconv_lstm_bagged(
            train_array, label_array, start_at=0, generator_batch_size=GENERATOR_BATCH_SIZE,use_precomputed_coeffs=False,
            scaled=True, scaler_type = 'standard_per_batch')
        num_generator_yields = train_array.shape[0]//GENERATOR_BATCH_SIZE
        base_model = Model(inputs=raw_base_model.input, outputs=raw_base_model.get_layer(name='dense_post_concat').output)
        base_model_output = base_model.predict_generator(xgb_train_generator,
                                                         steps=num_generator_yields)
        print(type(base_model_output))
        print(base_model_output.shape) #1,128,64 (if steps=1) or
        # num_generator_yields,GENERATOR_BATCH_SIZE, num of neurons in dense_after_concat
        base_model_output_dmatrix_shape = (base_model_output.shape[0] * base_model_output.shape[1], base_model_output.shape[2])
        base_model_output_dmatrix = np.zeros(shape=base_model_output_dmatrix_shape)
        for i in range(0,base_model_output.shape[0]):
            base_model_output_dmatrix[i:i+GENERATOR_BATCH_SIZE,:] = np.reshape() #the chunk
            i = i + GENERATOR_BATCH_SIZE


        data_dmatrix = xgb.DMatrix(data=base_model_output) #input to the DMatrix has to be 2D. default is 3D.
        #label_array_reshaped = np.reshape(label_array,newshape=()) #reshape to what? it needed reshaping for the Keras LSTM.
        label_dmatrix = xgb.DMatrix(data=label_array) #forget trying to get it through the generator.

        param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
        '''(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1,
        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)'''

        if i == 0:
            xgb_regressor = xgb.XGBRegressor(param)
            xgb_regressor_cached = xgb.XGBRegressor(param)
        if i != 0:
            xgb_regressor = xgb_regressor_cached
        xgb_regressor.fit(X=data_dmatrix,y=label_dmatrix)

    print(xgb_regressor.best_score)
        # training_hist = model.fit_generator(train_generator, steps_per_epoch=active_seq_circumnav_amt * (
        # train_array.shape[0] // generator_batch_size),
        #                                     epochs=num_epochs, verbose=2,
        #                                     callbacks=[csv_logger, nan_terminator])


        # aborted manual for for outer_counter in range (0,num_generator_yields):
        #     for inner_counter in range(0,128):
        #         generator_chunk

    #set as feature extractor.
    # g = concatenate(tensors_to_concat, name='concat_all')
    # h = BatchNormalization()(g)
    # i = Dense(64, activation=da, kernel_regularizer=kr, name='dense_post_concat')(h)

    #need the entire loop for loading the dataset.


    #initialize generator

    #gen_chunk_features
    #initialize xgboost
    # specify parameters via map, definition are same as c++ version

    #  #params as dicts?
    # xgb_notscikit = xgb.XGBModel()
    #
    # #use the model as feature extractor

    #
    # xgb_regressor.fit()



    #64-layer dense
    #
    # WAIT FOR THE NEW PARTIAL FIT API.
    # xgb_input features = base_model.predict_on_batch(gen_chunk_features)
    # dmatrix_input = xgb.DMatrix(gen_chunk_features)
    # #set up the data DMatrix.
    # #if counter == 1
    # xgb_regressor_trained = xgb_regressor.fit()
    # xgb_notscikit.fit()
    #
    # #postmortem
    # xgb_regressor_trained