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
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

#from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged
from LSTM_TimeDist import pair_generator_lstm
from AuxRegressor import create_training_set,create_testing_set,create_model_list,generate_model_id

'''This loads a saved Keras model and uses it as a feature extractor, which then feeds into Scikit-learn multivariate regressors. 
The generators created need to match what the Keras model asks for.'''

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
lstm_models_path = analysis_path + "lstm_models_to_load/"
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def create_lstm_model_list():
    # load models
    model_filenames = list(set(os.listdir(lstm_models_path)))
    # print("before sorting, data_filenames: {}".format(data_filenames))
    model_filenames.sort(reverse=False)
    # print("after sorting, data_filenames: {}".format(data_filenames))

    return model_filenames

if __name__ == "__main__":

    num_sequence_draws = 20
    GENERATOR_BATCH_SIZE = 128
    num_epochs = 1 #because you gotta feed the base model the same way you fed it during training... (RNN commandments)
    save_preds = True

    #create the data-pair filenames (using zip), use the helper methods
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    lstm_model_filenames = create_lstm_model_list()
    print(train_set_filenames)
    print(test_set_filenames)
    print(lstm_model_filenames)


    # load model
    #identifier shouldn't have a leading underscore!
    #TODO!!!!! REVERT BACK
    for lstm_model in lstm_model_filenames:
        identifier_post_training = lstm_model
        #identifier_post_training = "bag_conv_lstm_dense_tiny_shufstart_softplus_ca_tanh_da_3_cbd_standard_per_batch_sclr_l1l2_kr_HLR.h5"
        # './' + identifier_post_training + '.h5'
        print('loading model: ',lstm_models_path + lstm_model)
        raw_base_model = load_model(lstm_models_path + lstm_model)
        cond_conv = any(isinstance(layer, Conv1D) for layer in raw_base_model.layers)
        cond_lstm = any(isinstance(layer, LSTM) for layer in raw_base_model.layers)
        cond_bidir = any(isinstance(layer, Bidirectional) for layer in raw_base_model.layers)
        cond_has_dpc = any(layer.name == 'dense_post_concat' for layer in raw_base_model.layers)
        print("cond_lstm: ",cond_lstm, "cond_bidir: ",cond_bidir, "cond_has_dpc: ",cond_has_dpc, "cond_conv: ",cond_conv)
        #assert cond_conv == False and cond_lstm == True and cond_has_dpc == True
        time_dict = {}

        #switch to load item in that model_filenames list.
        print("using: {} as model".format(lstm_model))

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


            aux_reg_train_generator = pair_generator_lstm(
                train_array, label_array, start_at=0, generator_batch_size=GENERATOR_BATCH_SIZE,use_precomputed_coeffs=False,
                scaled=True, scaler_type = 'standard_per_batch',no_labels=True)
            num_generator_yields = train_array.shape[0]//GENERATOR_BATCH_SIZE
            #TODO: make this jump out if there is no dense_post_concat.
            #TODO: ALL THIS STUFF BELOW!!!!!!


            base_model = Model(inputs=raw_base_model.input, outputs=raw_base_model.get_layer(name='time_distributed_1').output)
            #timedist is a wrapper, so the original layer's name (dense_post_concat) is obscured
            base_model_output = base_model.predict_generator(aux_reg_train_generator,
                                                             steps=num_generator_yields)
            #print(type(base_model_output))
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
            #print(type(base_model_output_2d), type(label_array))
            print("for fitting: feature shape: {}, uncut label shape: {}, CUT label shape: {}".format(base_model_output_2d.shape, label_array.shape, label_array_to_fit.shape))

            if i == 0: #initialize for the first time
                #aux_reg_regressor = Ridge(solver='saga')
                #aux_reg_regressor = LinearRegression()
                #aux_reg_regressor = KernelRidge(alpha=1,kernel='polynomial',gamma=1.0e-3,)
                aux_reg_regressor = ExtraTreesRegressor(n_estimators=5,criterion='mse',n_jobs=2,warm_start=True)
                #aux_reg_regressor = RandomForestRegressor(n_estimators=30,criterion='mse',n_jobs=-1,warm_start=True)
                #aux_reg_regressor = MultiOutputRegressor(estimator=ElasticNet(warm_start=True), n_jobs=1)

                model_id = generate_model_id(aux_reg_regressor)
                assert model_id != ""
                print("model id is: ", model_id)

                print("fitting regressor..")

                aux_reg_regressor.fit(X=base_model_output_2d, y=label_array_to_fit)

                if isinstance(aux_reg_regressor,ExtraTreesRegressor) or isinstance(aux_reg_regressor,RandomForestRegressor):
                    tree_regressor_check_cond = True
                if not isinstance(aux_reg_regressor,ExtraTreesRegressor) and not isinstance(aux_reg_regressor,RandomForestRegressor):
                    tree_regressor_check_cond = False

            if i != 0:
                #aux_reg_regressor = aux_reg_regressor_cached
                #label_array_to_fit = label_scaler_aux_regressor.fit_transform(label_array[0:base_model_output_2d.shape[0],:])
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
        score_rows_list = []
        scores_dict = {}
        mse_dict = {}
        mae_dict = {}
        mape_dict = {}

        scores_dict_f3 = {}
        mse_dict_f3 = {}
        mae_dict_f3 = {}
        mape_dict_f3 = {}
        test_start_time = time.clock()

        getparams_dict = aux_reg_regressor.get_params(deep=True)
        print("getparams_dict: ", getparams_dict)
        getparams_df = pd.DataFrame.from_dict(data=getparams_dict,orient='index')
        getparams_df.to_csv(analysis_path + model_id + str(lstm_model)[:-4] + "getparams.csv")
        model_as_pkl_filename = analysis_path + model_id + str(lstm_model)[:-4] + ".pkl"
        joblib.dump(aux_reg_regressor,filename=model_as_pkl_filename)
        #np.savetxt(analysis_path + "rf5getparams.txt",fmt='%s',X=str(aux_reg_regressor.get_params(deep=True)))
        #np.savetxt(analysis_path + "rf5estimatorparams.txt",fmt='%s',X=aux_reg_regressor.estimator_params) USELESS
        #np.savetxt(analysis_path + "rf5classes.txt",fmt='%s',X=aux_reg_regressor.classes_)
        #np.savetxt(analysis_path + "rf5baseestim.txt",fmt='%s',X=aux_reg_regressor.base_estimator_)

        #TODO: CHANGE THIS BACK IF CUT SHORT!!
        for files in combined_filenames:
            print("filename", files)
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

            test_generator = pair_generator_lstm(
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

            index_last_3_batches = batch_scaled_test_labels.shape[0] - 3 * GENERATOR_BATCH_SIZE

            score = aux_reg_regressor.score(X=base_model_output_2d_test,y=test_label_array_to_fit)
            score_f3 = aux_reg_regressor.score(X=base_model_output_2d_test[index_last_3_batches:, :],
                                               y=test_label_array_to_fit[index_last_3_batches:, :])

            print("score: {}".format(score))
            print("score_f3: {}".format(score_f3))
            scores_dict[str(files[0])[:-4]] = score
            scores_dict_f3[str(files[0])[:-4]] = score_f3

            preds = aux_reg_regressor.predict(base_model_output_2d_test)
            if  save_preds == True:
                #<class 'sklearn.ensemble.forest.RandomForestRegressor'>
                #< class 'sklearn.ensemble.forest.ExtraTreesRegressor'>
                preds_filename = analysis_path + "preds_" + model_id + "_" + str(files[0])[:-4] + "_" + str(lstm_model)[:-3]
                np.save(file=preds_filename, arr=preds)
            mse_score = mean_squared_error(test_label_array_to_fit,preds)
            mse_score_f3 = mean_squared_error(test_label_array_to_fit[index_last_3_batches:, :],
                                              preds[index_last_3_batches:, :])
            print("mse: {}".format(mse_score))
            print("mse_f3: {}".format(mse_score_f3))
            mse_dict[str(files[0])[:-4]] = mse_score
            mse_dict_f3[str(files[0])[:-4]] = mse_score_f3

            mae_score = mean_absolute_error(test_label_array_to_fit,preds)
            mae_score_f3 = mean_absolute_error(test_label_array_to_fit[index_last_3_batches:, :],
                                              preds[index_last_3_batches:, :])
            print("mae: {}".format(mae_score))
            print("mae_f3: {}".format(mae_score_f3))
            mae_dict[str(files[0])[:-4]] = mae_score
            mae_dict_f3[str(files[0])[:-4]] = mae_score_f3
    test_end_time = time.clock()
    test_time_elapsed = test_end_time - test_start_time
    print("test time elapsed: {}".format(test_time_elapsed))
    time_dict['test_time'] = test_time_elapsed

    time_df = pd.DataFrame.from_dict(time_dict,orient='index')
    #time_df.rename(columns=['time'])
    r2_scores_df = pd.DataFrame.from_dict(scores_dict,orient='index')
    #r2_scores_df.rename(columns=['r2'])
    mse_scores_df = pd.DataFrame.from_dict(mse_dict,orient='index')
    #mse_scores_df.rename(columns=['mse'])
    mae_scores_df = pd.DataFrame.from_dict(mae_dict,orient='index')
    #mae_scores_df.rename(columns=['mae'])
    name_list = ['filename','r2','mse','mae']
    scores_combined_df = pd.DataFrame(pd.concat([r2_scores_df,mse_scores_df,mae_scores_df],axis=1))
    scores_combined_df.set_axis(labels=name_list[1:], axis=1)

    # time_df.to_csv("./analysis/time_rf5a_" + str(model) + ".csv")
    # r2_scores_df.to_csv("./analysis/r2_rf5a_" + str(model) + ".csv")
    # mse_scores_df.to_csv("./analysis/mse_rf5a_" + str(model) + ".csv")
    # mae_scores_df.to_csv("./analysis/mae_rf5a_" + str(model) + ".csv")
    scores_combined_df.to_csv("./analysis/combi_scores_" + model_id + "_" + str(lstm_model)[:-3] + str(num_sequence_draws) + "sd.csv")
