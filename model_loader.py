from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Dense, Dropout, \
    Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, concatenate, BatchNormalization
from keras.initializers import lecun_normal, glorot_normal,orthogonal
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
from Conv1D_ActivationSearch_BigLoop import pair_generator_1dconv_lstm #this is the stacked one.
from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged
from LSTM_TimeDist import pair_generator_lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, \
    r2_score, mean_squared_log_error

from AuxRegressor import create_model_list,create_testing_set,create_training_set, mape
# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
models_path = analysis_path + "models_to_load/"
results_path = analysis_path + "model_loader_results/"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def individual_output_scorer(metrics_list):
    score_df = pd.DataFrame() #set index
    return score_df

def create_generator(model_type, data, labels, start_at = 0,scaled = True,
                     type_scaler = 'standard_per_batch', gbs = 128,
                     upc = False, gen_pad = 128, lab_dim = 4):
    if model_type == 'conv_lstm_bagged' and cond_has_dpc == True:
        # create conv-lstm generator
        train_generator = pair_generator_1dconv_lstm_bagged(train_array, label_array, start_at=active_starting_position,
                                                            generator_batch_size=GENERATOR_BATCH_SIZE,
                                                            use_precomputed_coeffs=use_precomp_sscaler,
                                                            scaled=scaler_active, scaler_type=active_scaler_type,
                                                            label_dims=4, generator_pad=GENERATOR_PAD)
    if model_type == 'normal_lstm' or model_type == 'bidir_LSTM':
        # create lstm-only generator
        train_generator = pair_generator_lstm(train_array, label_array, start_at=shuffled_starting_position,
                                              generator_batch_size=GENERATOR_BATCH_SIZE,
                                              use_precomputed_coeffs=False, label_dims=4)
    if model_type == 'conv':
        train_generator = pair_generator_1dconv_lstm(train_array, label_array,
                                                     start_at=active_starting_position,
                                                     generator_batch_size=GENERATOR_BATCH_SIZE,
                                                     use_precomputed_coeffs=use_precomp_sscaler,
                                                     scaled=scaler_active,
                                                     scaler_type=active_scaler_type, label_dims=4,
                                                     generator_pad=GENERATOR_PAD)
    if model_type == 'conv':
        generator = pair_generator_1dconv_lstm(data=data,labels=labels,start_at=start_at,scaled=scaled,scaler_type=type_scaler,
                                               generator_batch_size=gbs, use_precomputed_coeffs = upc, label_dims=lab_dim)
    return generator
# def determine_state():
#     return state
#
# class State:

if __name__ == "__main__":

############################## RUNNING PARAMETERS #########################################################################
    num_sequence_draws = 10  # how many times the training corpus is sampled.
    GENERATOR_BATCH_SIZE = 128
    GENERATOR_PAD = 128 #for the conv-only and conv-LSTM generator.
    num_epochs = 3  # individual. like how many times is the net trained on that sequence consecutively
    finetune = True
    test_only = False  # no training. if finetune is also on, this'll raise an error.
    scaler_active = True
    use_precomp_sscaler = False
    active_scaler_type = 'standard_per_batch'
    if active_scaler_type != "None":
        assert (scaler_active != False)  # makes sure that if a scaler type is specified, the "scaler active" flag is on (the master switch)

    base_seq_circumnav_amt = 1.0  # default value, the only one if adaptive circumnav is False
    adaptive_circumnav = True
    if adaptive_circumnav == True:
        aux_circumnav_onset_draw = 3
        assert (aux_circumnav_onset_draw < num_sequence_draws)
        aux_seq_circumnav_amt = 1.5  # only used if adaptive_circumnav is True
        assert (base_seq_circumnav_amt != None and aux_seq_circumnav_amt != None and aux_circumnav_onset_draw != None)

    shuffle_training_generator = False
    shuffle_testing_generator = False #****

    save_preds = False
    save_figs = False


#######################################################################################################
# LOAD MODEL AND CHECK
    metrics_list = ['mae', 'mape', 'mse', 'msle']
    train_set_filenames = create_training_set()
    test_set_filenames = create_testing_set()
    model_filenames = create_model_list()

    #TODO: THIS BELOW!!!!
    for model in model_filenames:
        identifier_post_training = model
        raw_base_model = load_model(models_path + model)
        print("model loaded.")

        cond_conv = any(isinstance(layer, Conv1D) for layer in raw_base_model.layers)
        cond_lstm = any(isinstance(layer, LSTM) for layer in raw_base_model.layers)
        cond_bidir = any(isinstance(layer, Bidirectional) for layer in raw_base_model.layers)
        cond_has_dpc = any(layer.name == 'dense_post_concat' for layer in raw_base_model.layers)

        print("is there an LSTM layer?", any(isinstance(layer, LSTM) for layer in raw_base_model.layers))
        print("is there dpc?", any(layer.name == 'dense_post_concat' for layer in raw_base_model.layers))

        # cond_conv_lstm_bagged = cond_conv == True and cond_lstm == True
        if cond_lstm == True and cond_conv == True:
            model_type = 'conv_lstm_bagged'
        if cond_lstm == True and cond_conv == False:
            model_type = 'normal_lstm'
        if cond_lstm == True and cond_conv == False and cond_bidir == True:
            model_type = 'bidir_lstm'
        if cond_conv == True and cond_lstm == False:
            model_type = 'conv'

        #should be any item in metrics_list not in raw_base_model_metrics. any(metric not in raw_base_model.metrics for metric in metrics_list)
        cond_mismatched_metrics = any(metric not in raw_base_model.metrics for metric in metrics_list)
        #cond_not_multioutput = len(raw_base_model.metrics) >
        if cond_mismatched_metrics == True: #e.g. missing mse, recompile using the existing optimizers.
            print("mismatched metrics, model's metrics are currently: ",raw_base_model.metrics)
            existing_optimizer = raw_base_model.optimizer
            existing_loss = raw_base_model.loss
            raw_base_model.compile(optimizer = existing_optimizer,loss=existing_loss,metrics=metrics_list)

        print("model type is: ",model_type, " with metrics: ", raw_base_model.metrics_names)
        plot_model(raw_base_model, to_file=analysis_path + 'model_' + identifier_post_training + '_' + str(model)[:-4] + '.png', show_shapes=True)


#####################################################################################################

        weights_file_name = None
        identifier_post_training = 'f1' #placeholder for weights
        identifier_pre_training = 'f2' #placeholder, for weights
        if finetune == False:
            weights_present_indicator = os.path.isfile(
                'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
            print("Are weights (with the given name to be saved as) already present? {}".format(
                weights_present_indicator))
        else:
            weights_present_indicator = os.path.isfile('Weights_' + identifier_pre_training + '.h5')
            print("Are weights (with the given name) to initialize with present? {}".format(weights_present_indicator))
            if model is not None:
                weights_present_indicator = True
                print("model loaded instead, using: ", str(model))

        assert (finetune == False and weights_present_indicator == True) == False

        #initialize callbacks
        csv_logger = CSVLogger(filename='./analysis/logtrain' + identifier_post_training + ".csv", append=True)
        nan_terminator = TerminateOnNaN()
        active_seq_circumnav_amt = base_seq_circumnav_amt
############ TRAINING SET AND LABEL LOADS INTO MEMORY ###################################
        for i in range(0, num_sequence_draws):
            index_to_load = np.random.randint(0, len(train_set_filenames))  # switch to iterations
            files = train_set_filenames[index_to_load]
            print("files: {}, draw # {} out of {}".format(files, i,num_sequence_draws))
            data_load_path = train_path + '/data/' + files[0]
            label_load_path = train_path + '/label/' + files[1]

            train_array = np.load(data_load_path)
            if train_array.shape[1] != 11: #cut off the 1st column, which is the stepindex just for rigidity
                train_array = train_array[:, 1:]
            label_array = np.load(label_load_path)
            if label_array.shape[1] != 4: #cut off the 1st column, which is the stepindex just for rigidity
                label_array = label_array[:,1:]

            #TODO:if shuffle_training_generator == True:

            nonlinear_part_starting_position = GENERATOR_BATCH_SIZE * ((train_array.shape[0] // GENERATOR_BATCH_SIZE) - 3)
            shuffled_starting_position = np.random.randint(0, nonlinear_part_starting_position)

            if shuffle_training_generator == True:
                active_starting_position = shuffled_starting_position #doesn't start from 0, if the model is still in the 1st phase of training
            if shuffle_training_generator == False:
                active_starting_position = 0

            #adaptive circumnav governs the training from one generator upon initialization.
            if adaptive_circumnav == True and i >= aux_circumnav_onset_draw: #overrides the shuffle.
                active_seq_circumnav_amt = aux_seq_circumnav_amt
                active_starting_position = 0

            if model_type == 'conv_lstm_bagged' and cond_has_dpc == True:
                #create conv-lstm generator
                train_generator = pair_generator_1dconv_lstm_bagged(train_array, label_array, start_at=active_starting_position,generator_batch_size=GENERATOR_BATCH_SIZE,use_precomputed_coeffs=use_precomp_sscaler,
                                                                    scaled=scaler_active,scaler_type=active_scaler_type,label_dims=4,generator_pad=GENERATOR_PAD)
                # training_hist = raw_base_model.fit_generator(train_generator,
                #                                     steps_per_epoch=active_seq_circumnav_amt * (train_array.shape[0] // GENERATOR_BATCH_SIZE),
                #                                     epochs=num_epochs, verbose=2,
                #                                     callbacks=[csv_logger, nan_terminator])
            if model_type == 'normal_lstm' or model_type == 'bidir_LSTM':
                # create lstm-only generator
                train_generator = pair_generator_lstm(train_array, label_array, start_at=shuffled_starting_position,
                                                      generator_batch_size=GENERATOR_BATCH_SIZE,
                                                      use_precomputed_coeffs=False, label_dims=4)
                # training_hist = raw_base_model.fit_generator(train_generator, epochs=num_epochs,
                #                                     steps_per_epoch=1 * (train_array.shape[0] // GENERATOR_BATCH_SIZE),
                #                                     callbacks=[csv_logger, nan_terminator], verbose=2)

            if model_type == 'conv':
                train_generator = pair_generator_1dconv_lstm(train_array, label_array,
                                                             start_at=active_starting_position,
                                                             generator_batch_size=GENERATOR_BATCH_SIZE,
                                                             use_precomputed_coeffs=use_precomp_sscaler,
                                                             scaled=scaler_active,
                                                             scaler_type=active_scaler_type, label_dims=4,
                                                             generator_pad=GENERATOR_PAD)
                # training_hist = raw_base_model.fit_generator(train_generator, steps_per_epoch=active_seq_circumnav_amt * (train_array.shape[0] // GENERATOR_BATCH_SIZE),
                #                                     epochs=num_epochs, verbose=2, callbacks=[csv_logger, nan_terminator])

            training_hist = raw_base_model.fit_generator(train_generator,
                                                steps_per_epoch=active_seq_circumnav_amt * (train_array.shape[0] // GENERATOR_BATCH_SIZE),
                                                epochs=num_epochs, verbose=2,
                                                callbacks=[csv_logger, nan_terminator])
            trained_model = raw_base_model

        if weights_present_indicator == True and finetune == True:
            print("fine-tuning/partial training session completed.")
            weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
            #trained_model.save_weights(analysis_path + weights_file_name)
            trained_model.save(results_path + 'model_' + identifier_post_training + '_' + str(model)[:-3] + '.h5')
            print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                             weights_file_name))
        if weights_present_indicator == False and finetune == False:  # fresh training
            print("FRESH training session completed.")
            weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
            #trained_model.save_weights(weights_file_name)
            trained_model.save(results_path + 'model_' + identifier_post_training + '_' + str(model)[:-3] + '.h5')
            print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                             weights_file_name))
        else:  # TESTING ONLY! bypass weights present indicator.
            weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
            # test_weights_present_indicator

        print("weights_file_name before the if/else block to determine the test flag is: {}".format(
            weights_file_name))
        if weights_file_name is not None:
            # means it went through the training loop
            if os.path.isfile(weights_file_name) == False:
                print("Weights from training weren't saved as .h5 but is retained in memory.")
                test_weights_present_indicator = True
                print("test_weights_present_indicator is {}".format(test_weights_present_indicator))
                weights_to_test_with_fname = "weights retained in runtime memory"
            if os.path.isfile(weights_file_name) == True:
                test_weights_present_indicator = True
                print("test weights present indicator based on the presence of {} is {}".format(weights_file_name,
                                                                                                test_weights_present_indicator))
                weights_to_test_with_fname = weights_file_name
                model.load_weights(weights_to_test_with_fname, by_name=True)
        if test_only == True:
            trained_model = raw_base_model
            weights_to_test_with_fname = 'Weights_' + identifier_pre_training + '.h5'  # hardcode the previous epoch number UP ABOVE
            weights_file_name = weights_to_test_with_fname  # piggybacking the old flag. the one without fname is to refer to post training weights.
            trained_model.load_weights(weights_to_test_with_fname, by_name=True)
            test_weights_present_indicator = os.path.isfile(weights_to_test_with_fname)
        if weights_file_name == None:
            print(
                "Warning: check input flags. No training has been done, and testing is about to be performed with weights labeled as POST TRAINING weights")
            test_weights_present_indicator = os.path.isfile(
                'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
        print(
            "weights_file_name after the if/else block to determine the test flag is: {}".format(weights_file_name))

        if test_weights_present_indicator == True:
            # the testing part
            print("TESTING PHASE, with weights {}".format(weights_to_test_with_fname))

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

            i = 0
            # TODO: still only saves single results.
            score_rows_list = []
            score_rows_list_scikit = []
            score_rows_list_scikit_raw = []
            for files in combined_test_filenames:
                i = i + 1
                data_load_path = test_path + '/data/' + files[0]
                label_load_path = test_path + '/label/' + files[1]
                # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
                test_array = np.load(data_load_path)
                if test_array.shape[1] != 11:  # cut off the 1st column, which is the stepindex just for rigidity
                    test_array = test_array[:, 1:]
                label_array = np.load(label_load_path)
                if label_array.shape[1] != 4:  # cut off the 1st column, which is the stepindex just for rigidity
                    label_array = label_array[:, 1:]

                # --------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
                # test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
                # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
                # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
                print("sequence being tested: ",files[0], ", number ", i, "out of ", len(combined_test_filenames))
                # print("Metrics: {}".format(model.metrics_names))
                # steps per epoch is how many times that generator is called


                nonlinear_part_starting_position = GENERATOR_BATCH_SIZE * ((test_array.shape[0] // GENERATOR_BATCH_SIZE) - 3)
                shuffled_starting_position = np.random.randint(0, nonlinear_part_starting_position)

                if shuffle_testing_generator == True:
                    active_starting_position = shuffled_starting_position  # doesn't start from 0, if the model is still in the 1st phase of training
                if shuffle_testing_generator == False:
                    active_starting_position = 0

                #define the test generator parameters.

                if model_type == 'conv_lstm_bagged' and cond_has_dpc == True:
                    # create conv-lstm generator
                    test_generator = pair_generator_1dconv_lstm_bagged(test_array, label_array, start_at=active_starting_position,
                                                                       generator_batch_size=GENERATOR_BATCH_SIZE,
                                                                       use_precomputed_coeffs=use_precomp_sscaler,
                                                                       scaled=scaler_active,
                                                                       scaler_type=active_scaler_type,label_dims=4)
                if model_type == 'normal_lstm' or model_type == 'bidir_LSTM':
                    # create lstm-only generator
                    test_generator = pair_generator_lstm(test_array, label_array,
                                                          start_at=active_starting_position,
                                                          generator_batch_size=GENERATOR_BATCH_SIZE,
                                                          use_precomputed_coeffs=False, label_dims=4)
                if model_type == 'conv':
                    test_generator = pair_generator_1dconv_lstm(test_array, label_array,
                                                                 start_at=active_starting_position,
                                                                 generator_batch_size=GENERATOR_BATCH_SIZE,
                                                                 use_precomputed_coeffs=use_precomp_sscaler,
                                                                 scaled=scaler_active,
                                                                 scaler_type=active_scaler_type, label_dims=4,
                                                                 generator_pad=GENERATOR_PAD)
                score = trained_model.evaluate_generator(test_generator,
                                                 steps=(test_array.shape[0] // GENERATOR_BATCH_SIZE),
                                                 max_queue_size=test_array.shape[0], use_multiprocessing=False)


                print("scores: {}".format(score))

                metrics_check = (metrics_list == trained_model.metrics_names)
                if metrics_check == False:
                    metrics_list = trained_model.metrics_names

                row_dict = {}
                row_dict_scikit = {} #for the scikit-based metrics.
                row_dict_scikit_raw = {} #for the raw-value scikit-based metrics

                row_dict['seq_name'] = str(files[0])[:-4]
                for item in metrics_list:
                    row_dict[str(item)] = score[metrics_list.index(item)]  # 'loss'
                score_rows_list.append(row_dict)
                
                #SECOND TIME FOR PREDICTIONS LOGGING. INITIALIZE GENERATORS
                active_starting_position = 0 #hardcode for the actual predictions logging?
                if model_type == 'conv_lstm_bagged' and cond_has_dpc == True:
                    # create conv-lstm generator
                    test_generator = pair_generator_1dconv_lstm_bagged(test_array, label_array, start_at=active_starting_position,
                                                                       generator_batch_size=GENERATOR_BATCH_SIZE,
                                                                       use_precomputed_coeffs=use_precomp_sscaler,
                                                                       scaled=scaler_active,
                                                                       scaler_type=active_scaler_type, label_dims=4)
                if model_type == 'normal_lstm' or model_type == 'bidir_LSTM':
                    # create lstm-only generator
                    test_generator = pair_generator_lstm(test_array, label_array,
                                                         start_at=active_starting_position,
                                                         generator_batch_size=GENERATOR_BATCH_SIZE,
                                                         use_precomputed_coeffs=False, label_dims=4)
                if model_type == 'conv':
                    #create conv-only generator
                    test_generator = pair_generator_1dconv_lstm(test_array, label_array,
                                                                start_at=active_starting_position,
                                                                generator_batch_size=GENERATOR_BATCH_SIZE,
                                                                use_precomputed_coeffs=use_precomp_sscaler,
                                                                scaled=scaler_active,
                                                                scaler_type=active_scaler_type, label_dims=4,
                                                                generator_pad=GENERATOR_PAD)
                prediction_length = (int(1.0 * (GENERATOR_BATCH_SIZE * (label_array.shape[0] // GENERATOR_BATCH_SIZE))))
                test_i = 0
                # Kindly declare the shape
                preds_ndims = len(trained_model.output_layers[0].output_shape) #if 3D, this should be 3. [0] because this is the combined output.
                preds_2d_shape = [prediction_length, 4]
                preds_3d_shape = [1, prediction_length, 4]
                if preds_ndims == 3:
                    y_pred = np.zeros(shape = preds_3d_shape)
                    y_truth = np.zeros(shape = preds_3d_shape)
                if preds_ndims == 2:
                    y_pred = np.zeros(shape=preds_2d_shape)
                    y_truth = np.zeros(shape=preds_2d_shape)


                while test_i <= prediction_length - GENERATOR_BATCH_SIZE: #TODO: make sure this matches. [0] not necessary for bagged data, necessary for bagged labels.
                    x_test_batch, y_test_batch = test_generator.next()
                    if model_type == 'conv_lstm_bagged': #this has two outputs
                        y_pred[0, test_i:test_i + GENERATOR_BATCH_SIZE, :] = (trained_model.predict_on_batch(x_test_batch))[0]
                    if model_type != 'conv_lstm_bagged':
                        y_pred[0, test_i:test_i + GENERATOR_BATCH_SIZE, :] = trained_model.predict_on_batch(x_test_batch) #needs the entire output. #needs the entire output.
                    if model_type == 'conv_lstm_bagged': #duplicated outputs. pick the first one.
                        y_truth[0, test_i:test_i + GENERATOR_BATCH_SIZE, :] = y_test_batch[0]
                    if model_type != 'conv_lstm_bagged':
                        y_truth[0, test_i:test_i + GENERATOR_BATCH_SIZE, :] = y_test_batch
                    test_i += GENERATOR_BATCH_SIZE
                # print("array shape {}".format(y_prediction[0,int(0.95*prediction_length), :].shape))

                #gotta remove a dimension if the predictions are 3D, otherwise the scikit metrics won't work.
                if len(y_pred.shape) == 3:
                    y_pred = np.reshape(y_pred, newshape = preds_2d_shape)
                    y_truth = np.reshape(y_truth, newshape = preds_2d_shape)

                ind_f3 = y_pred.shape[0] - 3 * GENERATOR_BATCH_SIZE

                row_dict_scikit['seq_name'] = str(files[0])[:-4]
                row_dict_scikit_raw['seq_name'] = str(files[0])[:-4]
                row_dict_scikit['mse'] = mean_squared_error(y_true = y_truth, y_pred = y_pred,multioutput = 'uniform_average')
                row_dict_scikit['mse_f3'] = mean_squared_error(y_true=y_truth[ind_f3:,:], y_pred=y_pred[ind_f3:,:],
                                                            multioutput='uniform_average')
                raw_mse = list(mean_squared_error(y_true=y_truth, y_pred=y_pred,multioutput='raw_values'))
                for flaw in range(0,len(raw_mse)):
                    row_dict_scikit_raw['mse_' + str(flaw)] = raw_mse[flaw]
                raw_mse_f3 = list(mean_squared_error(y_true=y_truth[ind_f3:,:], y_pred=y_pred[ind_f3:,:],multioutput='raw_values'))
                for flaw in range(0, len(raw_mse_f3)):
                    row_dict_scikit_raw['mse_f3_' + str(flaw)] = raw_mse_f3[flaw]

                row_dict_scikit['mae'] = mean_absolute_error(y_true = y_truth, y_pred = y_pred,multioutput = 'uniform_average')
                row_dict_scikit['mae_f3'] = mean_absolute_error(y_true=y_truth[ind_f3:,:], y_pred=y_pred[ind_f3:,:],
                                                            multioutput='uniform_average')
                raw_mae = list(mean_absolute_error(y_true=y_truth, y_pred=y_pred,multioutput='raw_values'))
                for flaw in range(0,len(raw_mae)):
                    row_dict_scikit_raw['mae_' + str(flaw)] = raw_mae[flaw]
                raw_mae_f3 = list(mean_absolute_error(y_true=y_truth[ind_f3:,:], y_pred=y_pred[ind_f3:,:],multioutput='raw_values'))
                for flaw in range(0, len(raw_mae_f3)):
                    row_dict_scikit_raw['mae_f3_' + str(flaw)] = raw_mae_f3[flaw]
                # row_dict_scikit['msle'] = mean_squared_log_error(y_true=y_truth, y_pred=y_pred, multioutput='uniform_average')
                # row_dict_scikit['msle_f3'] = mean_squared_log_error(y_true=y_truth[ind_f3:,:], y_pred=y_pred[ind_f3:,:],
                #                                             multioutput='uniform_average')

                score_rows_list_scikit.append(row_dict_scikit)
                score_rows_list_scikit_raw.append(row_dict_scikit_raw)

                #print('row_dict sk keys: ', row_dict_scikit.keys(), "row dict sk values: ", row_dict_scikit.values())
                if save_preds == True:
                    np.save(analysis_path + 'preds/preds_' + identifier_post_training + str(files[0])[:-4] + '.npy',
                            arr=y_pred)

                # y_prediction_temp = y_truth
                # y_truth = np.reshape(y_truth,
                #                      newshape=(y_prediction_temp.shape[1], y_prediction_temp.shape[2]))
                # label_truth = label_array[0:y_truth.shape[0], :]
                # # print (label_truth.shape)
                # label_truth_temp = label_truth
                # scaler_output = sklearn.preprocessing.StandardScaler()  # TODO: this should use the precomputed coeffs as well...
                # #scaler_output = set_standalone_scaler_params(scaler_output)
                # # print("")
                # label_truth = scaler_output.transform(X=label_truth_temp)
                #
                # resample_interval = 16
                # label_truth = label_truth[::resample_interval, :]
                # y_truth = y_truth[::resample_interval, :]
            score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())
            score_scikit_df = pd.DataFrame(data=score_rows_list_scikit,columns=score_rows_list_scikit[0].keys())
            score_scikit_raw_df = pd.DataFrame(data=score_rows_list_scikit_raw,columns=score_rows_list_scikit_raw[0].keys())

            if shuffle_testing_generator == False:
                score_df.to_csv(analysis_path + 'scores_' + model_type + '_' + str(model)[:-3] + '.csv')
                score_scikit_df.to_csv(analysis_path + 'scores_sk_' + model_type + '_' + str(model)[:-3] + '.csv')
                score_scikit_raw_df.to_csv(analysis_path + 'scores_sk_raw_' + model_type + '_' + str(model)[:-3] + '.csv')

            if shuffle_testing_generator == True:
                score_df.to_csv(analysis_path + 'scores_' + model_type + '_' + str(model)[:-3] + 'shf_test.csv')
                score_scikit_df.to_csv(analysis_path + 'scores_sk_' + model_type + '_' + str(model)[:-3] + 'shf_test.csv')
                score_scikit_raw_df.to_csv(analysis_path + 'scores_sk_' + model_type + '_' + str(model)[:-3] + 'shf_test.csv')
            # print(len(y_prediction))


                #move all this junk into the training loop.
        #TODO check if the API are actually the same. if not, update.
        #TODO: initialize csvlogger






            #create conv-only generator



    # model.compile(loss={'combined_output': 'mape', 'lstm_output': 'mse'},
    #               optimizer=optimizer_used, metrics=metrics_list)