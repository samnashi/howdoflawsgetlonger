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

from Conv1D_ActivationSearch_BigLoop import pair_generator_1dconv_lstm #this is the stacked one.
from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged
from LSTM_TimeDist import pair_generator_lstm

from AuxRegressor import create_model_list,create_testing_set,create_training_set

if __name__ == "__main__":
    Model = Model.load