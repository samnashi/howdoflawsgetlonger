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
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, \
    r2_score
from sklearn.kernel_ridge import KernelRidge
import time
from sklearn.externals import joblib
from sklearn.utils import check_array


from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged

'''fits the scikit regressors only. required extensive work on making sure it works with the generator'''

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
models_path = analysis_path + "models_to_load/"


def mape(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_training_set():
    '''loads the input features and the labels, returns a list'''
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


def generate_model_id(model_sklearn):
    # <class 'sklearn.ensemble.forest.RandomForestRegressor'>
    # < class 'sklearn.ensemble.forest.ExtraTreesRegressor'>
    estim_num = 0
    model_type = ""
    model_identifier = ""
    if isinstance(model_sklearn, RandomForestRegressor):
        model_type = "rf"
        dict = model_sklearn.get_params(deep=True)
        estim_num = int(dict['n_estimators'])
        model_type = model_type + str(estim_num)
    if isinstance(model_sklearn, ExtraTreesRegressor):
        model_type = "et"
        dict = model_sklearn.get_params(deep=True)
        estim_num = int(dict['n_estimators'])
        model_type = model_type + str(estim_num)
    if isinstance(model_sklearn, Ridge):
        model_type = "ridge"
        dict = model_sklearn.get_params(deep=True)
        solver_type = str(dict['solver'])
        model_type = model_type + solver_type
    model_identifier = model_type
    return str(model_identifier)


if __name__ == "__main__":

    num_sequence_draws = 50
    GENERATOR_BATCH_SIZE = 128
    num_epochs = 1  # because you gotta feed the base model the same way you fed it during training... (RNN commandments)
    save_preds = True

    # create the data-pair filenames (using zip), use the helper methods
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    model_filenames = create_model_list()
    print(train_set_filenames)
    print(test_set_filenames)
    print(model_filenames)

    # load model
    # identifier shouldn't have a leading underscore!
    # TODO!!!!! REVERT BACK
    #for model in model_filenames[0:1]:

    label_scaler_aux_regressor = StandardScaler()
    input_scaler_aux_regressor = StandardScaler()
    train_start_time = time.clock()
    tree_regressor_check_cond = False
    time_dict = {}
    #batch_pointer = 0
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

        # train_array_to_scale = train_array[batch_pointer:batch_pointer+GENERATOR_BATCH_SIZE,:] #PER BATCH
        # batch_pointer += GENERATOR_BATCH_SIZE
        # train_array_to_scale = train_array[batch_pointer:batch_pointer + GENERATOR_BATCH_SIZE, :]
        # train_array_scaled = input_scaler_aux_regressor.fit_transform(X=train_array_to_scale)
        
        batch_scaled_inputs = np.zeros(shape=(train_array.shape))#TODO CHECK THE // GENERATOR_BATCH_SIZE part
        for train_batch_scaler_counter in range(0, train_array.shape[0]):  # how many batches there are
            batch_scaled_inputs[train_batch_scaler_counter:train_batch_scaler_counter + GENERATOR_BATCH_SIZE, :] = \
                input_scaler_aux_regressor.fit_transform(
                    train_array[train_batch_scaler_counter:train_batch_scaler_counter + GENERATOR_BATCH_SIZE, :])
            train_batch_scaler_counter += GENERATOR_BATCH_SIZE
        train_array_to_fit = batch_scaled_inputs[0:train_array.shape[0], :]

        # batch-scale the target array. per batch size.
        batch_scaled_labels = np.zeros(shape=(label_array.shape)) #TODO CHECK THE // GENERATOR_BATCH_SIZE part
        for label_batch_scaler_counter in range(0, label_array.shape[0]):  # how many batches there are
            batch_scaled_labels[label_batch_scaler_counter:label_batch_scaler_counter + GENERATOR_BATCH_SIZE, :] = \
                label_scaler_aux_regressor.fit_transform(
                    label_array[label_batch_scaler_counter:label_batch_scaler_counter + GENERATOR_BATCH_SIZE, :])
            label_batch_scaler_counter += GENERATOR_BATCH_SIZE
        label_array_to_fit = batch_scaled_labels[0:label_array.shape[0], :]
        print("for fitting: feature shape: {}, uncut label shape: {}".format(train_array.shape,
                                                                             label_array.shape))

        if i == 0:  # initialize for the first time
            # label_array_to_fit = label_scaler_aux_regressor.fit_transform(label_array[0:base_model_output_2d.shape[0],
            #                                                               :])  # but this is scaling the whole label set, not per batch.
            aux_reg_regressor = Ridge(solver='cholesky') #'svd' 'cholesky'(done) #'sparse_cg' #'lsqr' #'sag' #'saga'
            # aux_reg_regressor = LinearRegression()
            # aux_reg_regressor = KernelRidge(alpha=1,kernel='polynomial',gamma=1.0e-3,)
            #aux_reg_regressor = ExtraTreesRegressor(n_estimators=10,criterion='mse',n_jobs=-1,warm_start=True)
            #aux_reg_regressor = RandomForestRegressor(n_estimators=10, criterion='mse', n_jobs=-1, warm_start=True)

            print("fitting regressor..")
            aux_reg_regressor.fit(X=train_array_to_fit, y=label_array_to_fit)
            if isinstance(aux_reg_regressor, ExtraTreesRegressor) or isinstance(aux_reg_regressor,
                                                                                RandomForestRegressor):
                tree_regressor_check_cond = True
            if not isinstance(aux_reg_regressor, ExtraTreesRegressor) and not isinstance(aux_reg_regressor,
                                                                                         RandomForestRegressor):
                tree_regressor_check_cond = False

        if i != 0:
            # aux_reg_regressor = aux_reg_regressor_cached
            # label_array_to_fit = label_scaler_aux_regressor.fit_transform(
            #     label_array[0:base_model_output_2d.shape[0], :])
            print("fitting regressor..")
            if tree_regressor_check_cond == True:
                print("feat_imp before fitting: {}".format(aux_reg_regressor.feature_importances_))
            aux_reg_regressor.fit(X=train_array_to_fit, y=label_array_to_fit)
            if tree_regressor_check_cond == True:
                print("feat_imp after fitting: {}".format(aux_reg_regressor.feature_importances_))
                # aux_reg_regressor_cached = aux_reg_regressor.fit(X=base_model_output_2d,y=label_array_to_fit)
                # assert aux_reg_regressor_cached.feature_importances_ != aux_reg_regressor.feature_importances_
                # aux_reg_regressor = aux_reg_regressor_cached
    if tree_regressor_check_cond == True:
        print("feat-imp: {}, estimators: {}, estimator params: {} ".format(
            aux_reg_regressor.feature_importances_, aux_reg_regressor.estimators_,
            aux_reg_regressor.estimator_params))

        train_end_time = time.clock()
        train_time_elapsed = train_end_time - train_start_time
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
    scores_dict = {}
    mse_dict = {}
    mae_dict = {}
    mape_dict = {}
    test_start_time = time.clock()

    # try to save the trees attribute
    model_id = generate_model_id(aux_reg_regressor)
    if tree_regressor_check_cond == True:
        print("aux_reg_regressor estimator_params", aux_reg_regressor.estimators_)
    getparams_dict = aux_reg_regressor.get_params(deep=True)
    print(getparams_dict)
    getparams_df = pd.DataFrame.from_dict(data=getparams_dict, orient='index')
    getparams_df.to_csv(analysis_path + model_id + "getparams.csv")
    model_as_pkl_filename = analysis_path + model_id + ".pkl"
    joblib.dump(aux_reg_regressor, filename=model_as_pkl_filename)
    # np.savetxt(analysis_path + "rf5getparams.txt",fmt='%s',X=str(aux_reg_regressor.get_params(deep=True)))
    # np.savetxt(analysis_path + "rf5estimatorparams.txt",fmt='%s',X=aux_reg_regressor.estimator_params) USELESS
    # np.savetxt(analysis_path + "rf5classes.txt",fmt='%s',X=aux_reg_regressor.classes_)
    # np.savetxt(analysis_path + "rf5baseestim.txt",fmt='%s',X=aux_reg_regressor.base_estimator_)

    # TODO: CHANGE THIS BACK!!
    for files in combined_filenames:
        print("filename", files)
        i += 1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        test_label_array = np.load(label_load_path)[:, 1:]
        if test_array.shape[1] != 11:
            test_array = test_array[:, 1:]
        # --------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        # test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
        print(files[0])
        # print("Metrics: {}".format(model.metrics_names))
        # steps per epoch is how many times that generator is called

        batch_scaled_inputs = np.zeros(shape=(test_array.shape))#TODO CHECK THE // GENERATOR_BATCH_SIZE part
        for test_batch_scaler_counter in range(0, test_array.shape[0]):  # how many batches there are
            batch_scaled_inputs[test_batch_scaler_counter:test_batch_scaler_counter + GENERATOR_BATCH_SIZE, :] = \
                input_scaler_aux_regressor.fit_transform(
                    test_array[test_batch_scaler_counter:test_batch_scaler_counter + GENERATOR_BATCH_SIZE, :])
            test_batch_scaler_counter += GENERATOR_BATCH_SIZE
        test_array_to_fit = batch_scaled_inputs[0:test_array.shape[0], :]

        batch_scaled_test_labels = np.zeros(shape=(test_label_array.shape))
        for label_batch_scaler_counter in range(0, test_label_array.shape[0]):  # how many batches there are
            batch_scaled_test_labels[label_batch_scaler_counter:label_batch_scaler_counter + GENERATOR_BATCH_SIZE, :] = \
                label_scaler_aux_regressor.fit_transform(
                    test_label_array[label_batch_scaler_counter:label_batch_scaler_counter + GENERATOR_BATCH_SIZE, :])
            label_batch_scaler_counter += GENERATOR_BATCH_SIZE

        print("for fitting: feature shape: {}, uncut label shape: {}".format(test_array_to_fit.shape,
                                                                             batch_scaled_test_labels.shape))

        # tested_regressor = pkl.loads(trained_regressor)
        test_label_array_to_fit = batch_scaled_test_labels[0:test_array_to_fit.shape[0], :]
        score = aux_reg_regressor.score(X=test_array_to_fit, y=test_label_array_to_fit)
        print("score: {}".format(score))
        scores_dict[str(files[0])] = score
        preds = aux_reg_regressor.predict(test_array_to_fit)
        if save_preds == True:
            # <class 'sklearn.ensemble.forest.RandomForestRegressor'>
            # < class 'sklearn.ensemble.forest.ExtraTreesRegressor'>
            preds_filename = analysis_path + "preds_" + model_id + "_" + str(files[0])
            np.save(file=preds_filename, arr=preds)
        mse_score = mean_squared_error(test_label_array_to_fit, preds)
        print("mse: {}".format(mse_score))
        mae_score = mean_absolute_error(test_label_array_to_fit, preds)
        mape_score = mape(test_label_array_to_fit, preds)
        print("mae: {}".format(mae_score))
        mse_dict[str(files[0])] = mse_score
        mae_dict[str(files[0])] = mae_score
        mape_dict[str(files[0])] = mape_score
    test_end_time = time.clock()
    test_time_elapsed = test_end_time - test_start_time
    print("test time elapsed: {}".format(test_time_elapsed))
    time_dict['test_time'] = test_time_elapsed

    time_df = pd.DataFrame.from_dict(time_dict, orient='index')
    # time_df.rename(columns=['time'])
    r2_scores_df = pd.DataFrame.from_dict(scores_dict, orient='index')
    # r2_scores_df.rename(columns=['r2'])
    mse_scores_df = pd.DataFrame.from_dict(mse_dict, orient='index')
    # mse_scores_df.rename(columns=['mse'])
    mae_scores_df = pd.DataFrame.from_dict(mae_dict, orient='index')
    # mae_scores_df.rename(columns=['mae'])
    mape_scores_df = pd.DataFrame.from_dict(mape_dict, orient = 'index')

    name_list = ['filename', 'r2', 'mse', 'mae','mape']
    scores_combined_df = pd.DataFrame(pd.concat([r2_scores_df, mse_scores_df, mae_scores_df,mape_scores_df], axis=1))
    scores_combined_df.set_axis(labels=name_list[1:], axis=1)

    # time_df.to_csv("./analysis/time_rf5a_" + str(model) + ".csv")
    # r2_scores_df.to_csv("./analysis/r2_rf5a_" + str(model) + ".csv")
    # mse_scores_df.to_csv("./analysis/mse_rf5a_" + str(model) + ".csv")
    # mae_scores_df.to_csv("./analysis/mae_rf5a_" + str(model) + ".csv")
    scores_combined_df.to_csv("./analysis/combi_scores_" + model_id + "_" + ".csv")
