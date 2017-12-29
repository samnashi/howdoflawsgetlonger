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
stats_tests_path = analysis_path + 'stats_tests/'
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if __name__ == "__main__":
    test_filenames = create_testing_set()
    i=0
    summary_stats_dict={}
    for sequence_pair in test_filenames:
        data_load_path = test_path + '/data/' + sequence_pair[0]
        label_load_path = test_path + '/label/' + sequence_pair[1]
        print('loading: ',str(sequence_pair[0]))
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
            master_dadn = np.vstack((master_dadn,dadn_to_plot))
            master_deltaK = np.vstack((master_deltaK,deltaK_to_plot))
        i +=1
        #load array
        #loglog
    # summary_stats_dict['dadn_mean'] = np.average(master_dadn,axis=None)
    # summary_stats_dict['dadn_median'] = np.median(master_dadn,axis=None)
    # summary_stats_dict['dadn_stdev'] = np.std(master_dadn,axis=None)
    # summary_stats_dict['dadn_min'] = np.min(master_dadn,axis=None)
    # summary_stats_dict['dadn_max'] = np.max(master_dadn,axis=None)
    #
    # summary_stats_dict['deltaK_mean'] = np.average(master_deltaK,axis=None)
    # summary_stats_dict['deltaK_median'] = np.median(master_deltaK,axis=None)
    # summary_stats_dict['deltaK_stdev'] = np.std(master_deltaK,axis=None)
    # summary_stats_dict['deltaK_min'] = np.min(master_deltaK,axis=None)
    # summary_stats_dict['deltaK_max'] = np.max(master_deltaK,axis=None)
    #
    # summ_stats_df = pd.Series(summary_stats_dict).to_frame()
    # summ_stats_df.to_csv(stats_tests_path + 'summ_stats_TEST_axisnone.csv')
    plt.clf()
    plt.close()
    plt.cla()
    #plt.title('dadn vs deltaK of ' + str(sequence_pair[0][:-4]))
    #plt.plot(master_deltaK[1],master_dadn[1],color='blue',linestyle='dashed',marker='x')
    fig = plt.scatter(y=master_deltaK, x=master_dadn)
    plt.title('dadn vs deltaK of the test set')
    plt.xlabel=('da/dn, in/cycle')
    plt.ylabel=('deltaK, ksi * sqrt(in)')
    plt.yscale('log')
    plt.xscale('log')
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((0,6.54280584759e-05,0,18.7953955231))
    # plt.ylim((0,6.54280584759e-05)) #dadn
    # plt.xlim((0,18.7953955231)) #K
    plt.grid(True)
    #plt.loglog()
    plt.tight_layout()
    #plt.loglog()
    #_ = plt.scatter(y=deltaK_to_plot,x=dadn_to_plot)
    #_ = plt.scatter(y=master_deltaK, x=master_dadn)
   # plt.show()
    plt.savefig(analysis_path + 'stats_tests/' + 'loglog_test_set_5' + '.png')
    # master_dadn_df = pd.DataFrame(master_dadn)
    # master_deltaK_df = pd.DataFrame(master_deltaK)
    #master_dadn_df.to_csv(stats_tests_path + 'train_dadn_df.csv')
    #master_deltaK_df.to_csv(stats_tests_path + 'train_deltaK_df.csv')
    # np.savetxt(X=master_dadn,fname=stats_tests_path + 'train_dadn.csv')
    # np.savetxt(X=master_deltaK,fname=stats_tests_path + 'train_deltaK.csv')

    # plt.savefig(analysis_path + 'stats_tests/'+ 'loglog_' + str(sequence_pair[0])[:-4] + '.png')