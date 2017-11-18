from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
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

def set_standalone_scaler_params(output_scaler):
    '''intended to scale the output of the model to the same scaler as during training.currently set to 1d.'''
    output_scaler.var_ = [1.1455965013546072e-11, 1.1571155303166357e-11, 4.3949048693992676e-11,
                          4.3967045763969097e-11]
    output_scaler.mean_ = [4.5771139469142714e-06, 4.9590312890501306e-06, 6.916592701282579e-06,
                           6.9171280743598655e-06]
    output_scaler.scale_ = [3.3846661598370483e-06, 3.4016400901868433e-06, 6.6294078690327e-06, 6.63076509642508e-06]
    return output_scaler

def reference_bilstm_micro(input_tensor,k_init=lecun_normal(seed=1337),k_reg=l1(), rec_reg=l1(), sf = False,imp = 2, dense_act = 'tanh'):
    '''reference SMALL BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(4, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    #l = BatchNormalization()(k)
    #out = Dense(4)(l)
    out = k
    return out

def reference_bilstm_small(input_tensor,k_init=lecun_normal(seed=1337),k_reg=l1(), rec_reg=l1(), sf = False,imp = 2, dense_act = 'tanh'):
    '''reference SMALL BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(16, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    #l = BatchNormalization()(k)
    #out = Dense(4)(l)
    out = k
    return out

def reference_bilstm_big(input_tensor,k_init=lecun_normal(seed=1337), k_reg=l1(),rec_reg=l1(), sf = False,imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    # l = BatchNormalization()(k)
    # out = Dense(4)(l)
    out = k
    return out

def reference_lstm_nodense_micro(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(32, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(32, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_nodense_tiny(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_nodense_medium(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2):
    '''reference BiLSTM with batchnorm and NO dense layers. Output is already batch-normed.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_nodense(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_dense(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

def reference_lstm_dense_micro(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(4, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

def reference_lstm_dense_tiny(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

def reference_lstm_dense_medium(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

def reference_lstm_dense_huge(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(400, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(400, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

def reference_gru_dense_micro(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf = False, imp = 2, dense_act = 'tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = GRU(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = GRU(16, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp,stateful=sf)(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(4, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    out = k
    return out

# ---------------------REALLY WIDE WINDOW---------------------------------------------------------------------------------
def conv_block_normal_param_count(input_tensor, conv_act='relu', dense_act='relu',k_reg=None,k_init='lecun_normal'):
    '''f means it's the normal param count branch'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(8, kernel_size=(65), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(16, kernel_size=(65), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)  # gives me 128x1
    #g = BatchNormalization()(d)
    #h = Dense(1, activation=dense_act)(g)
    return d

def conv_block_double_param_count(input_tensor, conv_act='relu', dense_act='relu',feature_weighting=4,k_reg=None,k_init='lecun_normal'):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(16, kernel_size=(65), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(65), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)  # gives me 128x1
    #g = BatchNormalization()(d)
    #h = Dense(feature_weighting, activation=dense_act)(g)
    return d

# -----------------------------------------------------------------------------------------------------------------------
def conv_block_3layers_normal_param_count(input_tensor, conv_act='relu', dense_act='relu',k_reg=None,k_init='lecun_normal'):
    '''f means it's the normal param count branch. Padding required: 128#reqbatchsize -(128 - (128-1)/1 + (2-1)/1) = 128'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(8, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(16, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(32, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    #g = BatchNormalization()(f)
    #h = Dense(1, activation=dense_act)(g)
    return f


def conv_block_3layers_double_param_count(input_tensor, conv_act='relu', dense_act='relu',feature_weighting=2,k_reg=None,k_init='lecun_normal'):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    #input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(16, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(64, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    #g = BatchNormalization()(f)
    #h = Dense(feature_weighting, activation=dense_act,kernel_initializer=k_init)(g)
    return f

# ---------------------------------------------------------------------------------------------------
def conv_block_3layers_normal_pc_flatten(input_tensor, conv_act='relu', dense_act='relu',k_reg=None,k_init='lecun_normal'):
    '''f means it's the normal param count branch. Padding required: 128#reqbatchsize -(128 - (128-1)/1 + (2-1)/1) = 128'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(8, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(16, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(32, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    g = BatchNormalization()(f)
    #h = Flatten()(g)
    return g


def conv_block_3layers_double_pc_flatten(input_tensor, conv_act='relu', dense_act='relu',feature_weighting=2,k_reg=None,k_init='lecun_normal'):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(16, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(64, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    g = BatchNormalization()(f)
    #h = Flatten()(g)
    return g

#-----------------------------------------------------------------------------------------------------------
def conv_block_3layers_normal_pc_micro(input_tensor, conv_act='relu', dense_act='relu',k_reg=None,k_init='lecun_normal'):
    '''f means it's the normal param count branch. Padding required: 128#reqbatchsize -(128 - (128-1)/1 + (2-1)/1) = 128'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(4, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(8, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(16, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    g = BatchNormalization()(f)
    #h = Flatten()(g)
    return g


def conv_block_3layers_double_pc_micro(input_tensor, conv_act='relu', dense_act='relu',feature_weighting=2,k_reg=None,k_init='lecun_normal'):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(8, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(16, kernel_size =(65), padding='valid',activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)
    e = BatchNormalization()(d)
    f = Conv1D(32, kernel_size=(33), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(e)  # gives me 128x1
    g = BatchNormalization()(f)
    #h = Flatten()(g)
    return g
# ---------------------NARROW WINDOW-------------------------------------------------------------------------------------
def conv_block_normal_param_count_narrow_window(input_tensor, conv_act='relu', dense_act='relu',k_reg=None,k_init='lecun_normal'):
    '''requires generator batch for this column to be increased by 14. 2 * (8-1) = 14. '''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(8, kernel_size=(17), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(16, kernel_size=(17), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)  # gives me 128x1
    #g = BatchNormalization()(d)
    #h = Dense(1, activation=dense_act,kernel_initializer=k_init)(g)
    return d

def conv_block_double_param_count_narrow_window(input_tensor, conv_act='relu', dense_act='relu',feature_weighting=2,k_reg=None,k_init='lecun_normal'):
    '''requires generator batch for this column to be increased by 28. (15-1) + 2 * (8-1) = 28'''
    input_tensor = BatchNormalization()(input_tensor)
    b = Conv1D(16, kernel_size=(17), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(17), padding='valid', activation=conv_act,kernel_regularizer=k_reg,kernel_initializer=k_init)(c)  # gives me 128x1
    #g = BatchNormalization()(d)
    #h = Dense(feature_weighting, activation=dense_act,kernel_initializer=k_init)(g)
    return d

# -----------------------------------------------------------------------------------------------------------------------

#TODO: separate label and data scalers!
#TODO: copy the data array twice? or have two inputs. 
def pair_generator_1dconv_lstm_bagged(data, labels, start_at=0, generator_batch_size=64, scaled=True, scaler_type='standard',
                               use_precomputed_coeffs=True,generator_pad = 128, no_labels = False):  # shape is something like 1, 11520, 11
    '''Custom batch-yielding generator for Scattergro Output. You need to feed it the numpy array after running "Parse_Individual_Arrays script
    data and labels are self-explanatory.
    Parameters:
        start_at: configures where in the arrays do the generator start yielding (to ensure an LSTM doesn't always start at the same place
        generator_batch_size: how many "rows" of the numpy array does the generator yield each time
        scaled: whether the output is scaled or not.
        scaler_type: which sklearn scaler to call.
        - standard_per_batch is similar to what a batchnorm layer would do.
        - standard_minmax uses standardscaler on the data, and minmax on the stepindex
        - minmax_labels_only applies minmax only on the labels (the data scaling is done by a batchnorm layer after the input layer)
        scale_what = either the data/label (the whole array), or the yield.'''
    if scaled==False:
        scaler_type="None"
    if scaled == True:
        if scaler_type == 'standard' or scaler_type == "standard_per_batch":
            scaler = sklearn.preprocessing.StandardScaler()
            scaler_step_index_only = sklearn.preprocessing.StandardScaler()
            label_scaler = sklearn.preprocessing.StandardScaler()
        elif scaler_type == 'minmax' or scaler_type == "minmax_per_batch":
            scaler = sklearn.preprocessing.MinMaxScaler()
            label_scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust' or scaler_type == 'robust_per_batch':
            scaler = sklearn.preprocessing.RobustScaler()
            label_scaler = sklearn.preprocessing.RobustScaler()
        elif scaler_type == 'standard_minmax':
            scaler = sklearn.preprocessing.StandardScaler()
            scaler_step_index_only = sklearn.preprocessing.MinMaxScaler()
            label_scaler = sklearn.preprocessing.StandardScaler()
        elif scaler_type == 'minmax_labels_only':
            label_scaler = sklearn.preprocessing.MinMaxScaler
        else:
            scaler = sklearn.preprocessing.StandardScaler()
            label_scaler = sklearn.preprocessing.StandardScaler()
            # print("scaled: {}, scaler_type: {}".format(scaled,scaler_type))

    if scaled==True and use_precomputed_coeffs == True and scaler_type=='standard':
        # lists as dummy variables first, seems like scikit flips when I pass in a list as an object attribute..
        scaler_var = [0.6925742052047087, 0.016133766659421164,
                      0.6923827778657753, 0.019533317182529104, 3.621591547512037, 0.03208850741829512,
                      3.621824029181443, 0.03209691975648252, 43.47286356045491, 43.472882235044786]
        scaler_mean = [8.648004880708694, 0.5050077150656151,
                       8.648146575144597, 1.2382993509098987, 9.737983474596277, 1.7792042443537548,
                       9.737976755677462, 1.9832900698119342, 7.859076582026868, 7.859102808059667]
        scaler_scale = [0.8322104332467292, 0.12701876498935566,
                        0.8320954139194466, 0.1397616441751066, 1.9030479624833518, 0.1791326531325183,
                        1.9031090429035966, 0.1791561323440605, 6.593395450028377, 6.5933968661870175]
        label_scaler.var_ = [1.1455965013546072e-11, 1.1571155303166357e-11, 4.3949048693992676e-11,
                             4.3967045763969097e-11]
        label_scaler.mean_ = [4.5771139469142714e-06, 4.9590312890501306e-06, 6.916592701282579e-06,
                              6.9171280743598655e-06]
        label_scaler.scale_ = [3.3846661598370483e-06, 3.4016400901868433e-06, 6.6294078690327e-06,
                               6.63076509642508e-06]
        step_index_to_fit = np.reshape(data[:, 0], newshape=(-1, 1))
        # print("the shape scikit is bitching about: {}, and after reshape: {}".format(data[:,0].shape, step_index_to_fit.shape))
        revised_step_index = scaler_step_index_only.fit_transform(X=step_index_to_fit)  # gotta fit transform since \
        # TODO: /usr/local/lib/python2.7/dist-packages/sklearn/preprocessing/data.py:586: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
        # it makes no sense to precomp the stepindex. reshape is because sklearn gives a warning about 1D arrays as data..
        scaler_var.insert(0,
                          scaler_step_index_only.var_)  # append the fitted stepindex params into the main scaler object instance's params.
        scaler.var_ = np.asarray(scaler_var, dtype='float32')  # cast as numpy array so scikit won't flip
        scaler_mean.insert(0, scaler_step_index_only.mean_)  # set as scaler object attribute
        scaler.mean_ = np.asarray(scaler_mean, dtype='float32')
        scaler_scale.insert(0, scaler_step_index_only.scale_)
        scaler.scale_ = np.asarray(scaler_scale, dtype='float32')
        # print("data scaler mean shape: {} var shape: {} scale shape: {}".format(len(scaler.mean_),len(scaler.var_),len(scaler.scale_)))
        data_scaled = scaler.transform(X=data)
        labels_scaled = label_scaler.transform(X=labels)
        revised_reshaped_step_index = np.reshape(revised_step_index, newshape=(data[:,0].shape[0]))
        data[:,0] = revised_reshaped_step_index
    if use_precomputed_coeffs == False and scaler_type != "standard_minmax" and scaled==True:
        data_scaled = scaler.fit_transform(X=data)
        labels_scaled = label_scaler.fit_transform(X=labels)
    if use_precomputed_coeffs == False and scaler_type == 'standard_minmax' and scaled==True:
        step_index_to_fit = np.reshape(data[:, 0], newshape=(-1, 1))
        revised_step_index = scaler_step_index_only.fit_transform(X=step_index_to_fit)
        data_scaled = scaler.fit_transform(X=data)
        labels_scaled = label_scaler.fit_transform(X=labels)
        revised_reshaped_step_index = np.reshape(revised_step_index, newshape=(data[:,0].shape[0]))
        data[:,0] = revised_reshaped_step_index
    if use_precomputed_coeffs == False and scaler_type == "minmax_labels_only":
        data_scaled = data
        labels_scaled = label_scaler.fit(X=labels)

        # --------i think expand dims is a lot less implicit, that's why i commented these out-------
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))
        # ----------------------------------------------------------------------------------------------
        # print("before expand dims: data shape: {}, label shape: {}".format(data_scaled.shape,labels_scaled.shape))
    if not scaled or scaler_type == "standard_per_batch" or scaler_type == 'minmax_per_batch' or ("_per_batch" in scaler_type) :
        data_scaled = data
        labels_scaled = labels

    data_scaled = np.expand_dims(data_scaled, axis=0)  # add 1 dimension in the 0th axis.
    labels_scaled = np.expand_dims(labels_scaled, axis=0)
    index = start_at
    while 1:  # for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        # print((data_scaled[:, index:index + generator_batch_size_valid_x1, 0]).shape)
        generator_batch_size_valid_x1 = generator_batch_size + generator_pad
        generator_batch_size_valid_x2 = generator_batch_size + generator_pad
        generator_batch_size_valid_x3 = generator_batch_size + generator_pad
        generator_batch_size_valid_x4 = generator_batch_size + generator_pad
        generator_batch_size_valid_x5 = generator_batch_size + generator_pad
        generator_batch_size_valid_x6 = generator_batch_size + generator_pad
        generator_batch_size_valid_x7 = generator_batch_size + generator_pad
        generator_batch_size_valid_x8 = generator_batch_size + generator_pad
        generator_batch_size_valid_x9 = generator_batch_size + generator_pad
        generator_batch_size_valid_x10 = generator_batch_size + generator_pad
        generator_batch_size_valid_x11 = generator_batch_size + generator_pad

        if data_scaled.shape[2] > 11:
            x_lstm = data_scaled[:, index:index + generator_batch_size, 1:]
        if data_scaled.shape[2] == 11:
            x_lstm = data_scaled[:, index:index + generator_batch_size, :] #lstm's input.

        x1 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x1, 0]),
                        newshape=(1, (generator_batch_size_valid_x1), 1))  # first dim = 0 doesn't work.
        x2 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x2, 1]),
                        newshape=(1, (generator_batch_size_valid_x2), 1))
        x3 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x3, 2]),
                        newshape=(1, (generator_batch_size_valid_x3), 1))
        x4 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x4, 3]),
                        newshape=(1, (generator_batch_size_valid_x4), 1))
        x5 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x5, 4]),
                        newshape=(1, (generator_batch_size_valid_x5), 1))
        x6 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x6, 5]),
                        newshape=(1, (generator_batch_size_valid_x6), 1))
        x7 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x7, 6]),
                        newshape=(1, (generator_batch_size_valid_x7), 1))
        x8 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x8, 7]),
                        newshape=(1, (generator_batch_size_valid_x8), 1))
        x9 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x9, 8]),
                        newshape=(1, (generator_batch_size_valid_x9), 1))
        x10 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x10, 9]),
                         newshape=(1, (generator_batch_size_valid_x10), 1))
        x11 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x11, 10]),
                         newshape=(1, (generator_batch_size_valid_x11), 1))
        y = (labels_scaled[:, index:index + generator_batch_size, :])
        # if generator won't yield the full batch in 3 iterations, then..
        if index + 3 * generator_batch_size < data_scaled.shape[1]:
            index = index + generator_batch_size
        else:  # reset. anywhere between 0 and length of dataset - 2*batch size.
            index = np.random.randint(low=0, high=(
                generator_batch_size * ((data_scaled.shape[1] - start_at) // generator_batch_size - 2)))
            # ----------------ENABLE THIS FOR DIAGNOSTICS----------------------
            # print("x_shape at reset: {}".format(x.shape))
        # print("after expand dims:: data shape: {}, x1 shape: {}, x type: {}, y type:{}".format(data_scaled.shape,x1.shape,type(x1),type(y)))
        # x = np.reshape(x,(1,x.shape[0],x.shape[1]))
        # y = np.reshape(y, (1, y.shape[0],y.shape[1]))
        # print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x1.shape, y.shape))
        # if (index == data_scaled.shape[1] - 512): print("index reached: {}".format(index))
        # print("x: {}, y: {}".format(x,y))
        # -------------------ENABLE THIS FOR DIAGNOSTICS-----------------------
        # print("index: {}".format(index))
        # if (x.shape[1] != generator_batch_size_valid and y.shape[1] != generator_batch_size_valid): return
        # if (x.shape[1] != generator_batch_size_valid and y.shape[1] != generator_batch_size_valid): raise StopIteration
        assert (x1.shape[1] == generator_batch_size_valid_x1)  # if it's not yielding properly, stop.
        assert (x2.shape[1] == generator_batch_size_valid_x2)
        assert (x3.shape[1] == generator_batch_size_valid_x3)
        assert (x4.shape[1] == generator_batch_size_valid_x4)
        assert (x5.shape[1] == generator_batch_size_valid_x5)
        assert (x6.shape[1] == generator_batch_size_valid_x6)
        assert (x7.shape[1] == generator_batch_size_valid_x7)
        assert (x8.shape[1] == generator_batch_size_valid_x8)
        assert (x9.shape[1] == generator_batch_size_valid_x9)
        assert (x10.shape[1] == generator_batch_size_valid_x10)
        assert (x11.shape[1] == generator_batch_size_valid_x11)
        assert (y.shape[1] == generator_batch_size)
        if scaler_type == "standard_per_batch" or scaler_type == "minmax_per_batch" or ("_per_batch" in scaler_type) \
                and no_labels == False:
            # x1s = scaler.fit_transform(X=x1)
            # x2s = scaler.fit_transform(X=x2)
            # x3s = scaler.fit_transform(X=x3)
            # x4s = scaler.fit_transform(X=x4)
            # x5s = scaler.fit_transform(X=x5)
            # x6s = scaler.fit_transform(X=x6)
            # x7s = scaler.fit_transform(X=x7)
            # x8s = scaler.fit_transform(X=x8)
            # x9s = scaler.fit_transform(X=x9)
            # x10s = scaler.fit_transform(X=x10)
            # x11s = scaler.fit_transform(X=x11)
            ys = label_scaler.fit_transform(X=np.reshape(y,newshape=(y.shape[1],y.shape[2]))) #this is what's actually needed. you can't add a batchnorm layer to labels in Keras.
            y_re_exp = np.reshape(ys,newshape = (y.shape))
            yield ([x_lstm,x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], [y_re_exp,y_re_exp])
        elif scaler_type == "standard_per_batch" or scaler_type == "minmax_per_batch" or ("_per_batch" in scaler_type) \
                and no_labels == True:
            yield ([x_lstm, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
        else:
            yield ([x_lstm, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], [y,y]) #two outputs, multiple inputs.


if __name__ == "__main__":

    param_dict_HLR = param_dict_MLR = param_dict_LLR = {} #initialize all 3 as blank dicts
    param_dict_list = []

    param_dict_HLR['BatchSize'] = [128,128,128,128]
    param_dict_HLR['FeatWeight'] = [2,2,2,2]
    #NARROW WINDOW: 32 pad. WIDE WINDOW: 128 pad.
    param_dict_HLR['GenPad'] = [128,128,128,128]
    param_dict_HLR['ConvAct']=['relu','softplus','relu','softplus']
    param_dict_HLR['DenseAct']=['tanh','tanh','tanh','tanh']
    param_dict_HLR['KernelReg']=[l1_l2(),l1_l2(),l1_l2(),l1_l2()]
    param_dict_HLR['ConvBlockDepth'] = [3,3,3,3]
    param_dict_HLR['id_pre'] = []
    # make sure "Weights" isn't actually in the id. it is just the identifier after all.
    param_dict_HLR['id_post'] = []
    param_dict_HLR['ScalerType'] = ['standard_per_batch','standard_per_batch','minmax_per_batch','minmax_per_batch']
    reg_id = "" #placeholder. Keras L1 or L2 regularizers are 1 single class. You have to use get_config()['l1'] to see whether it's L1, L2, or L1L2

    for z in range(0, len(param_dict_HLR['BatchSize'])): #come up with a
        if len(param_dict_HLR['id_pre']) < len(param_dict_HLR['BatchSize']):
            param_dict_HLR['id_pre'].append("HLR_" + str(z)) #just a placeholder
        #ca = conv activation, da = dense activation, cbd = conv block depth
        #bag_conv_lstm_nodense_medium_shufstart
        id_post_temp = "_xgb_testmodel_" + str(param_dict_HLR['ConvAct'][z]) + "_ca_" + str(param_dict_HLR['DenseAct'][z]) + "_da_" + \
            str(param_dict_HLR['ConvBlockDepth'][z]) + "_cbd_" + str(param_dict_HLR['ScalerType'][z]) + "_sclr_"
        if param_dict_HLR['KernelReg'][z] != None:
            if (param_dict_HLR['KernelReg'][z].get_config())['l1'] != 0.0 and (param_dict_HLR['KernelReg'][z].get_config())['l2'] != 0.0:
                reg_id = "l1l2"
            if (param_dict_HLR['KernelReg'][z].get_config())['l1'] != 0.0 and (param_dict_HLR['KernelReg'][z].get_config())['l2'] == 0.0:
                reg_id = "l1"
            if (param_dict_HLR['KernelReg'][z].get_config())['l1'] == 0.0 and (param_dict_HLR['KernelReg'][z].get_config())['l2'] != 0.0:
                reg_id = "l2"
            id_post = id_post_temp + reg_id + "_kr_HLR"
        if param_dict_HLR['KernelReg'][z] == None:
            id_post = id_post_temp + str(param_dict_HLR['KernelReg'][z]) + "_kr_HLR"
        #below is the initial gridsearch params.
        # id_post = "_" + str(param_dict_HLR['BatchSize'][z]) + "_FeatWeight_" + str(param_dict_HLR['FeatWeight'][z]) + "_GenPad_" + \
        #           str(param_dict_HLR['GenPad'][z]) + "_HLR"
        param_dict_HLR['id_post'].append(id_post)

    #check lengths after the idpre and idpost aren't blank anymore.
    for key in param_dict_HLR.keys():
        print("checking key {}".format(key))
        assert(len(param_dict_HLR[key]) == len(param_dict_HLR['BatchSize']))

    for z in range(0, len(param_dict_HLR['BatchSize'])):
        #********************* READ FROM THE PARAMETER DICT ********************************************************************
        gen_pad = param_dict_HLR['GenPad'][z]
        bs = param_dict_HLR['BatchSize'][z]
        fw = param_dict_HLR['FeatWeight'][z]
        id_pre = param_dict_HLR['id_pre'][z]
        id_post = param_dict_HLR['id_post'][z]
        cbd = param_dict_HLR['ConvBlockDepth'][z]
        kr = param_dict_HLR['KernelReg'][z]
        da = param_dict_HLR['DenseAct'][z]
        ca = param_dict_HLR['ConvAct'][z]
        st = param_dict_HLR['ScalerType'][z]

        # !!!!!!!!!!!!!!!!!!!! TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK THESE FLAGS YO!!!!!!!!!!!!
        # shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
        num_epochs = 2  # individual. like how many times is the net trained on that sequence consecutively
        num_sequence_draws = 10  # how many times the training corpus is sampled.
        generator_batch_size = bs
        finetune = False
        test_only = False  # no training. if finetune is also on, this'll raise an error.
        scaler_active = True
        use_precomp_sscaler = False
        active_scaler_type = st #no capitals!
        if active_scaler_type != "None":
            assert(scaler_active != False) #makes sure that if a scaler type is specified, the "scaler active" flag is on (the master switch)

        base_seq_circumnav_amt = 1.0 #default value, the only one if adaptive circumnav is False
        adaptive_circumnav = True
        if adaptive_circumnav == True:
            aux_circumnav_onset_draw = 5
            assert(aux_circumnav_onset_draw < num_sequence_draws)
            aux_seq_circumnav_amt = 1.2 #only used if adaptive_circumnav is True
            assert(base_seq_circumnav_amt != None and aux_seq_circumnav_amt != None and aux_circumnav_onset_draw != None)

        save_preds = True
        save_figs = False
        env = "blockade_runner"  # "cruiser" "chan" #TODO complete the environment_variable_setter
        generator_batch_size_valid_x1 = (generator_batch_size + gen_pad)#4layer conv
        generator_batch_size_valid_x2 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x3 = (generator_batch_size + gen_pad)#4layer conv
        generator_batch_size_valid_x4 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x5 = (generator_batch_size + gen_pad)#4layer conv
        generator_batch_size_valid_x6 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x7 = (generator_batch_size + gen_pad)#4layer conv
        generator_batch_size_valid_x8 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x9 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x10 = (generator_batch_size + gen_pad)
        generator_batch_size_valid_x11 = (generator_batch_size + gen_pad)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # identifier = "_convlstm_run1_" + str(generator_batch_size) + "b_completev1data_valid_4layer_1357_"
        # Weights_200_conv1d_samepad_1_128shortrun

        # "_conv1d_samepad_" + str(num_epochs) + "_" + str(generator_batch_size) + "shortrun"
        # Weights_200_conv1d_samepad_1_128shortrun
        identifier_post_training = id_post
        # identifier_pre_training = "_conv1d_samepad_" + str(num_epochs) + "_" + str(generator_batch_size) + "shortrun"
        identifier_pre_training = id_pre  # for now, make hardcode what you want to finetune
        # @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        Base_Path = "./"
        image_path = "./images/"
        train_path = "./train/"
        test_path = "./test/"
        analysis_path = "./analysis/"
        # ^^^^^^^^^^^^^ TO RUN ON CHEZ CHAN ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Base_Path = "/home/devin/Documents/PITTA LID/"
        # image_path = "/home/devin/Documents/PITTA LID/img/"
        # train_path = "/home/devin/Documents/PITTA LID/Train FV1b/"
        # test_path = "/home/devin/Documents/PITTA LID/Test FV1b/"
        # test_path = "/home/devin/Documents/PITTA LID/FV1b 1d nonlinear/"
        # +++++++++++++ TO RUN ON LOCAL (IHSAN) +++++++++++++++++++++++++++++++
        # Base_Path = "/home/ihsan/Documents/thesis_models/"
        # image_path = "/home/ihsan/Documents/thesis_models/images"
        # train_path = "/home/ihsan/Documents/thesis_models/train/"
        # test_path = "/home/ihsan/Documents/thesis_models/test/"
        # analysis_path = "/home/ihsan/Documents/thesis_models/analysis/"
        # %%%%%%%%%%%%% TO RUN ON LOCAL (EFI) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Base_Path = "/home/efi/Documents/thesis_models/"
        # image_path = "/home/efi/Documents/thesis_models/images"
        # train_path = "/home/efi/Documents/thesis_models/train/"
        # test_path = "/home/efi/Documents/thesis_models/test/"
        # analysis_path = "home/efi/Documents/thesis_models/analysis"
        # seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

        #two loops. 2 vs. 3 layers.
        if cbd == 2:
            if gen_pad == 128:
                print("cbd 2 and gen pad 128 chosen: ")
                g1 = conv_block_double_param_count(input_tensor=a1,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f2 = conv_block_normal_param_count(input_tensor=a2,k_reg=kr,conv_act=ca,dense_act=da)
                g3 = conv_block_double_param_count(input_tensor=a3,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f4 = conv_block_normal_param_count(input_tensor=a4,k_reg=kr,conv_act=ca,dense_act=da)
                g5 = conv_block_double_param_count(input_tensor=a5,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f6 = conv_block_normal_param_count(input_tensor=a6,k_reg=kr,conv_act=ca,dense_act=da)
                g7 = conv_block_double_param_count(input_tensor=a7,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f8 = conv_block_normal_param_count(input_tensor=a8,k_reg=kr,conv_act=ca,dense_act=da)
                f9 = conv_block_normal_param_count(input_tensor=a9,k_reg=kr,conv_act=ca,dense_act=da)
                f10 = conv_block_normal_param_count(input_tensor=a10,k_reg=kr,conv_act=ca,dense_act=da)
                f11 = conv_block_normal_param_count(input_tensor=a11,k_reg=kr,conv_act=ca,dense_act=da)
        if cbd == 3:
            if gen_pad == 128:
                # g1 = conv_block_3layers_double_param_count(input_tensor=a1,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                # f2 = conv_block_3layers_normal_param_count(input_tensor=a2,k_reg=kr,conv_act=ca,dense_act=da)
                # g3 = conv_block_3layers_double_param_count(input_tensor=a3,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                # f4 = conv_block_3layers_normal_param_count(input_tensor=a4,k_reg=kr,conv_act=ca,dense_act=da)
                # g5 = conv_block_3layers_double_param_count(input_tensor=a5,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                # f6 = conv_block_3layers_normal_param_count(input_tensor=a6,k_reg=kr,conv_act=ca,dense_act=da)
                # g7 = conv_block_3layers_double_param_count(input_tensor=a7,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                # f8 = conv_block_3layers_normal_param_count(input_tensor=a8,k_reg=kr,conv_act=ca,dense_act=da)
                # f9 = conv_block_3layers_normal_param_count(input_tensor=a9,k_reg=kr,conv_act=ca,dense_act=da)
                # f10 = conv_block_3layers_normal_param_count(input_tensor=a10,k_reg=kr,conv_act=ca,dense_act=da)
                # f11 = conv_block_3layers_normal_param_count(input_tensor=a11,k_reg=kr,conv_act=ca,dense_act=da)

                g1 = conv_block_3layers_double_pc_micro(input_tensor=a1,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f2 = conv_block_3layers_normal_pc_micro(input_tensor=a2,k_reg=kr,conv_act=ca,dense_act=da)
                g3 = conv_block_3layers_double_pc_micro(input_tensor=a3,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f4 = conv_block_3layers_normal_pc_micro(input_tensor=a4,k_reg=kr,conv_act=ca,dense_act=da)
                g5 = conv_block_3layers_double_pc_micro(input_tensor=a5,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f6 = conv_block_3layers_normal_pc_micro(input_tensor=a6,k_reg=kr,conv_act=ca,dense_act=da)
                g7 = conv_block_3layers_double_pc_micro(input_tensor=a7,feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
                f8 = conv_block_3layers_normal_pc_micro(input_tensor=a8,k_reg=kr,conv_act=ca,dense_act=da)
                f9 = conv_block_3layers_normal_pc_micro(input_tensor=a9,k_reg=kr,conv_act=ca,dense_act=da)
                f10 = conv_block_3layers_normal_pc_micro(input_tensor=a10,k_reg=kr,conv_act=ca,dense_act=da)
                f11 = conv_block_3layers_normal_pc_micro(input_tensor=a11,k_reg=kr,conv_act=ca,dense_act=da)
        else:
            print("else block chosen.")
            g1 = conv_block_double_param_count_narrow_window(input_tensor=a1, feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
            f2 = conv_block_normal_param_count_narrow_window(input_tensor=a2,k_reg=kr,conv_act=ca,dense_act=da)
            g3 = conv_block_double_param_count_narrow_window(input_tensor=a3, feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
            f4 = conv_block_normal_param_count_narrow_window(input_tensor=a4,k_reg=kr,conv_act=ca,dense_act=da)
            g5 = conv_block_double_param_count_narrow_window(input_tensor=a5, feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
            f6 = conv_block_normal_param_count_narrow_window(input_tensor=a6,k_reg=kr,conv_act=ca,dense_act=da)
            g7 = conv_block_double_param_count_narrow_window(input_tensor=a7, feature_weighting=fw,k_reg=kr,conv_act=ca,dense_act=da)
            f8 = conv_block_normal_param_count_narrow_window(input_tensor=a8,k_reg=kr,conv_act=ca,dense_act=da)
            f9 = conv_block_normal_param_count_narrow_window(input_tensor=a9,k_reg=kr,conv_act=ca,dense_act=da)
            f10 = conv_block_normal_param_count_narrow_window(input_tensor=a10,k_reg=kr,conv_act=ca,dense_act=da)
            f11 = conv_block_normal_param_count_narrow_window(input_tensor=a11,k_reg=kr,conv_act=ca,dense_act=da)
        # if you want to use causal: you REALLY need shape[1] (# of steps) to be explicit.

        # reference_bilstm outputs the 64-time-dist-dense unit.
        lstm_in = Input(shape=(None, 11),name='lstm_input')  # still need to mod the generator to not pad...

        lstm = reference_lstm_nodense_medium(input_tensor=lstm_in, k_reg=kr, k_init='orthogonal', rec_reg=kr)
        lstm_bn = BatchNormalization(name='final_bn')(lstm)
        lstm_out = Dense(4,name='lstm_output')(lstm)

        tensors_to_concat = [g1, f2, g3, f4, g5, f6, g7, f8, f9, f10, f11,lstm_out]
        g = concatenate(tensors_to_concat,name='concat_all')
        h = BatchNormalization()(g)
        i = Dense(64,activation=da,kernel_regularizer=kr,name='dense_post_concat')(h)
        j = BatchNormalization()(i)
        out = Dense(4,name='combined_output')(j)


        model = Model(inputs=[lstm_in, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], outputs=[out,lstm_out])
        plot_model(model, to_file=analysis_path + 'model_' + identifier_post_training + '.png', show_shapes=True)
        optimizer_used = rmsprop(lr=0.0059)
        # loss = {'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
        # loss_weights = {'main_output': 1., 'aux_output': 0.2})

        model.compile(loss={'combined_output': 'mape', 'lstm_output': 'mse'},
                      optimizer=optimizer_used, metrics=['accuracy', 'mae', 'mape', 'mse','msle'])
        print("Model summary: {}".format(model.summary()))

        print("Inputs: {}".format(model.input_shape))
        print("Outputs: {}".format(model.output_shape))
        print("Metrics: {}".format(model.metrics_names))

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
        weights_file_name = None

        if finetune == False:
            weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
            print("Are weights (with the given name to be saved as) already present? {}".format(weights_present_indicator))
        else:
            weights_present_indicator = os.path.isfile('Weights_' + identifier_pre_training + '.h5')
            print("Are weights (with the given name) to initialize with present? {}".format(weights_present_indicator))

        csv_logger = CSVLogger(filename='./analysis/logtrain' + identifier_post_training + ".csv", append=True)
        nan_terminator = TerminateOnNaN()
        active_seq_circumnav_amt = 0.0 #predeclare a float.

        if (finetune == False and weights_present_indicator == False and test_only == False) or (
                finetune == True and weights_present_indicator == True):
            print("TRAINING PHASE")
            print("weights_present_indicator: {}, finetune: {}".format(weights_present_indicator, finetune))

            active_seq_circumnav_amt = base_seq_circumnav_amt #active is the one given to the generator..
            #adaptive or not, it starts the same way (at the base rate)
            for i in range(0, num_sequence_draws):
                index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
                files = combined_filenames[index_to_load]
                print("files: {}".format(files))
                data_load_path = train_path + '/data/' + files[0]
                label_load_path = train_path + '/label/' + files[1]
                # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
                train_array = np.load(data_load_path)
                label_array = np.load(label_load_path)[:, 1:]
                if train_array.shape[1] != 11:
                    train_array = train_array[:,1:]
                print("data/label shape: {}, {}, draw #: {}".format(train_array.shape, label_array.shape, i))
                # train_array = np.reshape(train_array,(1,generator_batch_size,train_array.shape[1]))
                # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!


                nonlinear_part_starting_position = generator_batch_size * ((train_array.shape[0] // generator_batch_size) - 3)
                shuffled_starting_position = np.random.randint(0, nonlinear_part_starting_position)
                active_starting_position = shuffled_starting_position #doesn't start from 0, if the model is still in the 1st phase of training
                #active_starting_position = 0

                if adaptive_circumnav == True and i >= aux_circumnav_onset_draw:
                    active_seq_circumnav_amt = aux_seq_circumnav_amt
                    active_starting_position = 0


                if finetune == True:  # load the weights
                    finetune_init_weights_filename = 'Weights_' + identifier_pre_training + '.h5'  # hardcode the previous epoch number UP ABOVE
                    model.load_weights(finetune_init_weights_filename, by_name=True)

                train_generator = pair_generator_1dconv_lstm_bagged(train_array, label_array, start_at=active_starting_position,
                                                             generator_batch_size=generator_batch_size,
                                                             use_precomputed_coeffs=use_precomp_sscaler,scaled=scaler_active,
                                                             scaler_type=active_scaler_type)
                training_hist = model.fit_generator(train_generator, steps_per_epoch=active_seq_circumnav_amt * (train_array.shape[0] // generator_batch_size),
                                                    epochs=num_epochs, verbose=2,
                                                    callbacks=[csv_logger, nan_terminator])

            if weights_present_indicator == True and finetune == True:
                print("fine-tuning/partial training session completed.")
                weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
                model.save_weights(weights_file_name)
                model.save('./model_' + identifier_post_training + '.h5')
                print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                                 weights_file_name))
            if weights_present_indicator == False and finetune == False:  # fresh training
                print("FRESH training session completed.")
                weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
                model.save_weights(weights_file_name)
                model.save('./model_' + identifier_post_training + '.h5')
                print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                                 weights_file_name))
            else:  # TESTING ONLY! bypass weights present indicator.
                weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
                # test_weights_present_indicator

        print("weights_file_name before the if/else block to determine the test flag is: {}".format(weights_file_name))
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
            weights_to_test_with_fname = 'Weights_' + identifier_pre_training + '.h5'  # hardcode the previous epoch number UP ABOVE
            weights_file_name = weights_to_test_with_fname  # piggybacking the old flag. the one without fname is to refer to post training weights.
            model.load_weights(weights_to_test_with_fname, by_name=True)
            test_weights_present_indicator = os.path.isfile(weights_to_test_with_fname)
        if weights_file_name == None:
            print(
                "Warning: check input flags. No training has been done, and testing is about to be performed with weights labeled as POST TRAINING weights")
            test_weights_present_indicator = os.path.isfile(
                'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
        print("weights_file_name after the if/else block to determine the test flag is: {}".format(weights_file_name))

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
                test_generator = pair_generator_1dconv_lstm_bagged(test_array, label_array, start_at=0,
                                                            generator_batch_size=generator_batch_size,
                                                            use_precomputed_coeffs=use_precomp_sscaler,scaled=scaler_active,
                                                            scaler_type=active_scaler_type)

                score = model.evaluate_generator(test_generator, steps=(test_array.shape[0] // generator_batch_size),
                                                 max_queue_size=test_array.shape[0], use_multiprocessing=False)
                row_dict = {}
                print("scores: {}".format(score))

                #Metrics: ['loss', 'acc', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error']
                #for keys in model.metrics_names: #TODO: make the model read the metric names and have a dict as large as the metric names.

                row_dict['filename'] = str(files[0])[:-4]
                row_dict['loss'] = score[0]  # 'loss'
                row_dict['acc'] = score[1]  # 'acc'
                row_dict['mae'] = score[2]  # 'mean_absolute_error'
                row_dict['mape'] = score[3]  # 'mean_absolute_percentage_error'
                row_dict['mse'] = score[4]  # 'mean_absolute_percentage_error'
                score_rows_list.append(row_dict)

                # testing should start at 0. For now.
                # #initializing generator a second time, to save predictions.
                test_generator = pair_generator_1dconv_lstm_bagged(test_array, label_array, start_at=0,
                                                            generator_batch_size=generator_batch_size,
                                                            use_precomputed_coeffs=use_precomp_sscaler,scaled=scaler_active,
                                                            scaler_type=active_scaler_type)
                prediction_length = (int(1.0 * (generator_batch_size * (label_array.shape[0] // generator_batch_size))))
                test_i = 0
                # Kindly declare the shape
                x_prediction = np.zeros(shape=[1, prediction_length, 4])
                y_prediction = np.zeros(shape=[1, prediction_length, 4])

                while test_i <= prediction_length - generator_batch_size:
                    x_test_batch, y_test_batch = test_generator.next()
                    x_prediction[0, test_i:test_i + generator_batch_size, :] = model.predict_on_batch(x_test_batch)[0] #just the combined output
                    y_prediction[0, test_i:test_i + generator_batch_size, :] = y_test_batch[0]
                    test_i += generator_batch_size
                # print("array shape {}".format(y_prediction[0,int(0.95*prediction_length), :].shape))
                if save_preds == True:
                    np.save(analysis_path + 'preds/preds_' + identifier_post_training + str(files[0]), arr=y_prediction)

                # print(y_prediction.shape)
                # print (x_prediction.shape)
                # print ("label array shape: {}".format(label_array.shape))

                # print("y_prediction shape: {}".format(y_prediction.shape))
                y_prediction_temp = y_prediction
                y_prediction = np.reshape(y_prediction, newshape=(y_prediction_temp.shape[1], y_prediction_temp.shape[2]))
                label_truth = label_array[0:y_prediction.shape[0], :]
                # print (label_truth.shape)
                label_truth_temp = label_truth
                scaler_output = sklearn.preprocessing.StandardScaler()  # TODO: this should use the precomputed coeffs as well...
                scaler_output = set_standalone_scaler_params(scaler_output)
                # print("")
                label_truth = scaler_output.transform(X=label_truth_temp)

                resample_interval = 16
                label_truth = label_truth[::resample_interval, :]
                y_prediction = y_prediction[::resample_interval, :]

            score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())
            score_df.to_csv('scores_conv_' + identifier_post_training + '.csv')
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
            # DEVIN PLOT CODE
            # if save_figs == True:
            #     plt.clf()
            #     plt.cla()
            #     plt.close()
            #     plt.plot(label_truth[:,0],'^',label="ground truth", markersize=5)
            #     plt.plot(y_prediction[:,0],'.',label="prediction", markersize=4)
            #     plt.xscale('log')
            #     plt.xlabel('# Cycle(s)')
            #     plt.yscale('log')
            #     plt.ylabel('Value(s)')
            #     plt.legend()
            #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
            #     plt.title('truth vs prediction from 50% - 100% of the sequence on Crack 01')
            #     plt.grid(True)
            #     plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
            #
            #     plt.clf()
            #     plt.cla()
            #     plt.close()
            #     plt.plot(label_truth[:,1],'^',label="ground truth", markersize=5)
            #     plt.plot(y_prediction[:,1],'v',label="prediction", markersize=4)
            #     plt.xscale('log')
            #     plt.xlabel('# Cycle(s)')
            #     plt.yscale('log')
            #     plt.ylabel('Value(s)')
            #     plt.legend()
            #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
            #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 02')
            #     plt.grid(True)
            #     plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
            #
            #     plt.clf()
            #     plt.cla()
            #     plt.close()
            #     plt.plot(label_truth[:,2],'^',label="ground truth", markersize=5)
            #     plt.plot(y_prediction[:,2],'v',label="prediction", markersize=4)
            #     plt.xscale('log')
            #     plt.xlabel('# Cycle(s)')
            #     plt.yscale('log')
            #     plt.ylabel('Value(s)')
            #     plt.legend()
            #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
            #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 03')
            #     plt.grid(True)
            #     plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
            #
            #     plt.clf()
            #     plt.cla()
            #     plt.close()
            #     plt.plot(label_truth[:,3],'^',label="ground truth", markersize=5)
            #     plt.plot(y_prediction[:,3],'v',label="prediction", markersize=4)
            #     plt.xscale('log')
            #     plt.xlabel('# Cycle(s)')
            #     plt.yscale('log')
            #     plt.ylabel('Value(s)')
            #     plt.legend()
            #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
            #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 04')
            #     plt.grid(True)
            #     plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')