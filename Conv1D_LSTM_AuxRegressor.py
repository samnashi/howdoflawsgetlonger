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
#import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge

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

    num_sequence_draws = 20
    GENERATOR_BATCH_SIZE = 128
    num_epochs = 1 #because you gotta feed the base model the same way you fed it during training... (RNN commandments)

    #create the data-pair filenames (using zip), use the helper methods
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    print(train_set_filenames)
    print(test_set_filenames)

    # load model
    #identifier shouldn't have a leading underscore!
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

        aux_reg_train_generator = pair_generator_1dconv_lstm_bagged(
            train_array, label_array, start_at=0, generator_batch_size=GENERATOR_BATCH_SIZE,use_precomputed_coeffs=False,
            scaled=True, scaler_type = 'standard_per_batch')
        num_generator_yields = train_array.shape[0]//GENERATOR_BATCH_SIZE
        base_model = Model(inputs=raw_base_model.input, outputs=raw_base_model.get_layer(name='dense_post_concat').output)
        base_model_output = base_model.predict_generator(aux_reg_train_generator,
                                                         steps=num_generator_yields)
        print(type(base_model_output))
        print("base model output shape: {}".format(base_model_output.shape)) #1,128,64 (if steps=1) or
        # num_generator_yields,GENERATOR_BATCH_SIZE, num of neurons in dense_after_concat
        base_model_output_2d_shape = (base_model_output.shape[0] * base_model_output.shape[1], base_model_output.shape[2])
        base_model_output_2d = np.zeros(shape=base_model_output_2d_shape)
        for reshape_counter in range(0,base_model_output.shape[0]):
            base_model_output_2d[i:i + GENERATOR_BATCH_SIZE, :] = np.reshape(
                base_model_output[i,:,:],newshape=(GENERATOR_BATCH_SIZE,base_model_output.shape[2])) #1,128,64 to 128,64
            reshape_counter += GENERATOR_BATCH_SIZE
        print(base_model_output_2d.shape)

        #data_dmatrix = xgb.DMatrix(data=base_model_output_dmatrix) #input to the DMatrix has to be 2D. default is 3D.
        #label_array_reshaped = np.reshape(label_array,newshape=()) #reshape to what? it needed reshaping for the Keras LSTM.
        #label_dmatrix = xgb.DMatrix(data=label_array) #forget trying to get it through the generator.
        print(type(base_model_output_2d), type(label_array))
        print("for fitting: feature shape: {}, uncut label shape: {}".format(base_model_output_2d.shape, label_array.shape))

        if i == 0:
            # aux_reg_regressor = Ridge()
            # aux_reg_regressor = LinearRegression()
            # aux_reg_regressor = ExtraTreesRegressor()
            aux_reg_regressor = RandomForestRegressor(n_estimators=10,criterion='mae')

            # aux_reg_regressor_cached = Ridge()
            # aux_reg_regressor_cached = LinearRegression()
            # aux_reg_regressor_cached = ExtraTreesRegressor()
            aux_reg_regressor_cached = RandomForestRegressor()
        if i != 0:
            aux_reg_regressor = aux_reg_regressor_cached
        label_array_to_fit = label_array[0:base_model_output_2d.shape[0],:]
        print("fitting regressor..")
        aux_reg_regressor_cached = aux_reg_regressor.fit(X=base_model_output_2d,y=label_array_to_fit)
    #aux_reg_regressor.
    print("feat-imp: {}, estimators: {}, estimator params: {} ".format(
        aux_reg_regressor.feature_importances_,aux_reg_regressor.estimators_,aux_reg_regressor.estimator_params))

    print("TESTING PHASE")
    data_filenames = list(set(os.listdir(test_path + "data")))
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    label_filenames = list(set(os.listdir(test_path + "label")))
    label_filenames.sort()
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_filenames))
    shuffle(combined_filenames)
    print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.

    i = 0
    # TODO: still only saves single results.
    score_rows_list = []
    for files in combined_filenames:
        i = i + 1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        # --------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        # test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
        print(files[0])
        # print("Metrics: {}".format(model.metrics_names))
        # steps per epoch is how many times that generator is called

        test_generator = pair_generator_1dconv_lstm_bagged(
            train_array, label_array, start_at=0, generator_batch_size=GENERATOR_BATCH_SIZE,
            use_precomputed_coeffs=False,
            scaled=True, scaler_type='standard_per_batch')
        #aux_reg_regressor.score()
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
    # aux_reg_notscikit = xgb.XGBModel()
    #
    # #use the model as feature extractor

    #
    # aux_reg_regressor.fit()



    #64-layer dense
    #
    # WAIT FOR THE NEW PARTIAL FIT API.
    # aux_reg_input features = base_model.predict_on_batch(gen_chunk_features)
    # dmatrix_input = xgb.DMatrix(gen_chunk_features)
    # #set up the data DMatrix.
    # #if counter == 1
    # aux_reg_regressor_trained = aux_reg_regressor.fit()
    # aux_reg_notscikit.fit()
    #
    # #postmortem
    # aux_reg_regressor_trained