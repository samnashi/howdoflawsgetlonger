from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pickle as pkl
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, \
    r2_score
from sklearn.kernel_ridge import KernelRidge
import time

from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
models_path = analysis_path + "models_to_load/"

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

def create_model_list():
    # load models
    model_filenames = list(set(os.listdir(models_path)))
    # print("before sorting, data_filenames: {}".format(data_filenames))
    model_filenames.sort(reverse=False)
    # print("after sorting, data_filenames: {}".format(data_filenames))

    return model_filenames

if __name__ == "__main__":

    num_sequence_draws = 300
    GENERATOR_BATCH_SIZE = 128
    num_epochs = 1 #because you gotta feed the base model the same way you fed it during training... (RNN commandments)

    #create the data-pair filenames (using zip), use the helper methods
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    model_filenames = create_model_list()
    print(train_set_filenames)
    print(test_set_filenames)
    print(model_filenames)


    # load model
    #identifier shouldn't have a leading underscore!
    for model in model_filenames:
        identifier_post_training = model
        #identifier_post_training = "bag_conv_lstm_dense_tiny_shufstart_softplus_ca_tanh_da_3_cbd_standard_per_batch_sclr_l1l2_kr_HLR.h5"
        # './' + identifier_post_training + '.h5'
        raw_base_model = load_model(models_path + model)
        time_dict = {}

        #switch to load item in that model_filenames list.
        print("using: {} as model".format(model))

        label_scaler_aux_regressor = StandardScaler()
        train_start_time = time.clock()
        tree_regressor_check_cond = False
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
                scaled=True, scaler_type = 'standard_per_batch',no_labels=True)
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
                base_model_output_2d[reshape_counter:reshape_counter + GENERATOR_BATCH_SIZE, :] = np.reshape(
                    base_model_output[reshape_counter,:,:],newshape=(GENERATOR_BATCH_SIZE,base_model_output.shape[2])) #1,128,64 to 128,64
                reshape_counter += 1
            print("pretrained net's output shape: {}".format(base_model_output_2d.shape))

            #batch-scale the target array. per batch size.
            batch_scaled_labels = np.zeros(shape=(label_array.shape))
            for label_batch_scaler_counter in range(0,label_array.shape[0]): #how many batches there are
                batch_scaled_labels[label_batch_scaler_counter:label_batch_scaler_counter+GENERATOR_BATCH_SIZE,:] = \
                    label_scaler_aux_regressor.fit_transform(
                        label_array[label_batch_scaler_counter:label_batch_scaler_counter+GENERATOR_BATCH_SIZE,:])
                label_batch_scaler_counter += GENERATOR_BATCH_SIZE
            label_array_to_fit = batch_scaled_labels[0:base_model_output_2d.shape[0],:]
            #data_dmatrix = xgb.DMatrix(data=base_model_output_dmatrix) #input to the DMatrix has to be 2D. default is 3D.
            #label_array_reshaped = np.reshape(label_array,newshape=()) #reshape to what? it needed reshaping for the Keras LSTM.
            #label_dmatrix = xgb.DMatrix(data=label_array) #forget trying to get it through the generator.
            print(type(base_model_output_2d), type(label_array))
            print("for fitting: feature shape: {}, uncut label shape: {}".format(base_model_output_2d.shape, label_array.shape))

            if i == 0: #initialize for the first time
                # label_array_to_fit = label_scaler_aux_regressor.fit_transform(label_array[0:base_model_output_2d.shape[0],
                #                                                               :])  # but this is scaling the whole label set, not per batch.
                #aux_reg_regressor = Ridge()
                #aux_reg_regressor = LinearRegression()
                #aux_reg_regressor = KernelRidge(alpha=1,kernel='polynomial',gamma=1.0e-3,)
                #aux_reg_regressor = ExtraTreesRegressor(n_estimators=5,criterion='mse',n_jobs=2,warm_start=True)
                aux_reg_regressor = RandomForestRegressor(n_estimators=5,criterion='mse',n_jobs=2,warm_start=True)
                
                print("fitting regressor..")
                aux_reg_regressor.fit(X=base_model_output_2d, y=label_array_to_fit)
                if isinstance(aux_reg_regressor,ExtraTreesRegressor) or isinstance(aux_reg_regressor,RandomForestRegressor):
                    tree_regressor_check_cond = True
                if not isinstance(aux_reg_regressor,ExtraTreesRegressor) and not isinstance(aux_reg_regressor,RandomForestRegressor):
                    tree_regressor_check_cond = False

                # aux_reg_regressor_cached = Ridge()
                # aux_reg_regressor_cached = LinearRegression()
                #aux_reg_regressor_cached = ExtraTreesRegressor(n_estimators=20,criterion='mse',n_jobs=3)
                # aux_reg_regressor_cached = RandomForestRegressor(n_estimators=5,criterion='mse',n_jobs=3)
            if i != 0:
                #aux_reg_regressor = aux_reg_regressor_cached
                label_array_to_fit = label_scaler_aux_regressor.fit_transform(label_array[0:base_model_output_2d.shape[0],:])
                print("fitting regressor..")
                if tree_regressor_check_cond == True:
                    print("feat_imp before fitting: {}".format(aux_reg_regressor.feature_importances_))
                aux_reg_regressor.fit(X=base_model_output_2d, y=label_array_to_fit)
                if tree_regressor_check_cond == True:
                    print("feat_imp after fitting: {}".format(aux_reg_regressor.feature_importances_))
                # aux_reg_regressor_cached = aux_reg_regressor.fit(X=base_model_output_2d,y=label_array_to_fit)
                #assert aux_reg_regressor_cached.feature_importances_ != aux_reg_regressor.feature_importances_
                #aux_reg_regressor = aux_reg_regressor_cached
        if tree_regressor_check_cond == True:
            print("feat-imp: {}, estimators: {}, estimator params: {} ".format(
                aux_reg_regressor.feature_importances_,aux_reg_regressor.estimators_,aux_reg_regressor.estimator_params))

        train_end_time = time.clock()
        train_time_elapsed = train_start_time - train_end_time
        print("training time elapsed: {}".format(train_time_elapsed))
        time_dict['train'] = train_time_elapsed

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
        scores_dict={}
        mse_dict={}
        mae_dict={}
        test_start_time = time.clock()
        for files in combined_filenames:
            i += 1
            data_load_path = test_path + '/data/' + files[0]
            label_load_path = test_path + '/label/' + files[1]
            # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
            test_array = np.load(data_load_path)
            test_label_array = np.load(label_load_path)[:, 1:]
            # --------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
            # test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
            # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
            # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
            print(files[0])
            # print("Metrics: {}".format(model.metrics_names))
            # steps per epoch is how many times that generator is called

            test_generator = pair_generator_1dconv_lstm_bagged(
                test_array, test_label_array, start_at=0, generator_batch_size=GENERATOR_BATCH_SIZE,
                use_precomputed_coeffs=False,
                scaled=True, scaler_type='standard_per_batch',no_labels=True)

            num_generator_yields = test_array.shape[0]//GENERATOR_BATCH_SIZE
            base_model_output_test = base_model.predict_generator(test_generator,
                                                             steps=num_generator_yields)
            base_model_output_2d_test_shape = (base_model_output_test.shape[0] * base_model_output_test.shape[1], base_model_output_test.shape[2])
            base_model_output_2d_test = np.zeros(shape=base_model_output_2d_test_shape)
            reshape_counter_test=0
            for reshape_counter_test in range(0,base_model_output_test.shape[0]):
                base_model_output_2d_test[reshape_counter_test:reshape_counter_test + GENERATOR_BATCH_SIZE, :] = np.reshape(
                    base_model_output_test[reshape_counter_test,:,:],newshape=(GENERATOR_BATCH_SIZE,base_model_output_test.shape[2])) #1,128,64 to 128,64
                reshape_counter_test += 1

            batch_scaled_test_labels = np.zeros(shape=(test_label_array.shape))
            for label_batch_scaler_counter in range(0,base_model_output_test.shape[0]): #how many batches there are
                batch_scaled_test_labels[label_batch_scaler_counter:label_batch_scaler_counter+GENERATOR_BATCH_SIZE,:] = \
                    label_scaler_aux_regressor.fit_transform(
                        test_label_array[label_batch_scaler_counter:label_batch_scaler_counter+GENERATOR_BATCH_SIZE,:])
                label_batch_scaler_counter += GENERATOR_BATCH_SIZE

            print("for fitting: feature shape: {}, uncut label shape: {}".format(base_model_output_2d_test.shape,
                                                                                     batch_scaled_test_labels.shape))
            print("pretrained net's output shape: {}".format(base_model_output_2d_test.shape))

            #tested_regressor = pkl.loads(trained_regressor)
            test_label_array_to_fit = batch_scaled_test_labels[0:base_model_output_2d_test.shape[0],:]
            score = aux_reg_regressor.score(X=base_model_output_2d_test,y=test_label_array_to_fit)
            print("score: {}".format(score))
            scores_dict[str(files[0])] = score
            preds = aux_reg_regressor.predict(base_model_output_2d_test)
            mse_score = mean_squared_error(test_label_array_to_fit,preds)
            mae_score = mean_absolute_error(test_label_array_to_fit,preds)
            mse_dict[str(files[0])] = mse_score
            mae_dict[str(files[0])] = mae_score
        test_end_time = time.clock()
        test_time_elapsed = test_end_time - test_start_time
        print("test time elapsed: {}".format(test_time_elapsed))
        time_dict['test_time'] = test_time_elapsed

        time_df = pd.DataFrame.from_dict(time_dict,orient='index')
        r2_scores_df = pd.DataFrame.from_dict(scores_dict,orient='index')
        mse_scores_df = pd.DataFrame.from_dict(mse_dict,orient='index')
        mae_scores_df = pd.DataFrame.from_dict(mae_dict,orient='index')
        scores_combined_df = pd.DataFrame(pd.concat([r2_scores_df,mse_scores_df,mae_scores_df]))

        time_df.to_csv("./analysis/time_rf5_" + str(model) + ".csv")
        r2_scores_df.to_csv("./analysis/r2_rf5_" + str(model) + ".csv")
        mse_scores_df.to_csv("./analysis/mse_rf5_" + str(model) + ".csv")
        mae_scores_df.to_csv("./analysis/mae_rf5_" + str(model) + ".csv")
        scores_combined_df.to_csv("./analysis/combi_scores_rf5_" + str(model) + ".csv")
