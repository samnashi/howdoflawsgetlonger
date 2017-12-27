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

if __name__ == "__main__":
    test_filenames = create_training_set()
    i=0
    for sequence_pair in test_filenames:
        data_load_path = train_path + '/data/' + sequence_pair[0]
        label_load_path = train_path + '/label/' + sequence_pair[1]
        #    train_list = ['StepIndex','percent_damage','delta_K_current_1','ctip_posn_curr_1','delta_K_current_2','ctip_posn_curr_2',
          #'delta_K_current_3','ctip_posn_curr_3','delta_K_current_4','ctip_posn_curr_4','Load_1','Load_2']
        train_array = np.load(data_load_path)
        if train_array.shape[1] != 11:  # cut off the 1st column, which is the stepindex just for rigidity
            train_array = train_array[:, 1:]
        label_array = np.load(label_load_path)
        if label_array.shape[1] != 4:  # cut off the 1st column, which is the stepindex just for rigidity
            label_array = label_array[:, 1:]
        dadn_to_plot = label_array
        deltaK_to_plot = train_array[:,[1,3,5,7]]
        if i == 0:
            master_dadn = dadn_to_plot
            master_deltaK = deltaK_to_plot
        else:
            master_dadn = np.vstack(master_dadn,dadn_to_plot)
            master_deltaK = np.vstack(master_deltaK,deltaK_to_plot)
        #load array
        #loglog
    plt.clf()
    plt.close()
    #plt.title('dadn vs deltaK of ' + str(sequence_pair[0][:-4]))
    plt.title('dadn vs deltaK of the test set')
    plt.xlabel=('da/dn')
    plt.ylabel=('deltaK')
    plt.loglog()
    #_ = plt.scatter(y=deltaK_to_plot,x=dadn_to_plot)
    _ = plt.scatter(y=master_deltaK, x=master_dadn)
   # plt.show()
    i =+1
    plt.savefig(analysis_path + 'stats_tests/' + 'loglog_train_set' + '.png')
    # plt.savefig(analysis_path + 'stats_tests/'+ 'loglog_' + str(sequence_pair[0])[:-4] + '.png')