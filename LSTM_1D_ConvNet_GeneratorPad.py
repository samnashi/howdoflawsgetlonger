from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Dense, Dropout, \
    Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, concatenate, BatchNormalization
from keras.initializers import lecun_normal,glorot_normal
from keras import metrics
from keras.optimizers import adam
import pandas as pd
import scipy.io as sio
from keras.callbacks import CSVLogger
import os
import csv
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing
# from LSTM_1D_ConvNet_Base import pair_generator_1dconv_lstm, conv_block_normal_param_count, conv_block_double_param_count, \
#     conv_block_normal_param_count_narrow_window, conv_block_double_param_count_narrow_window,\
#     conv_block_double_param_count_narrow_window_causal, conv_block_normal_param_count_narrow_window_causal, reference_bilstm

# np.set_printoptions(threshold='nan')

def set_standalone_scaler_params(output_scaler):
    '''intended to scale the output of the model to the same scaler as during training.currently set to 1d.'''
    output_scaler.var_ = [1.1455965013546072e-11, 1.1571155303166357e-11, 4.3949048693992676e-11, 4.3967045763969097e-11]
    output_scaler.mean_ = [4.5771139469142714e-06, 4.9590312890501306e-06, 6.916592701282579e-06, 6.9171280743598655e-06]
    output_scaler.scale_ = [3.3846661598370483e-06, 3.4016400901868433e-06, 6.6294078690327e-06, 6.63076509642508e-06]
    return output_scaler

def reference_bilstm(input_tensor):
    '''reference BiLSTM with batchnorm and elu TD-dense. Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(500, kernel_initializer=lecun_normal(seed=1337), return_sequences=True))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(500, kernel_initializer=lecun_normal(seed=1337), return_sequences=True))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=lecun_normal(seed=1337), activation='selu'))(j)
    l = BatchNormalization()(k)
    out = Dense(4)(l)
    return out

def reference_bigru(input_tensor):
    '''reference BiLSTM with batchnorm and elu TD-dense. Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(GRU(200, kernel_initializer=lecun_normal(seed=1337), return_sequences=True))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(GRU(200, kernel_initializer=lecun_normal(seed=1337), return_sequences=True))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=lecun_normal(seed=1337), activation='elu'))(j)
    l = BatchNormalization()(k)
    out = Dense(4)(l)
    return out

#---------------------REALLY WIDE WINDOW---------------------------------------------------------------------------------
def conv_block_normal_param_count(input_tensor,conv_activation = 'selu', dense_activation = 'elu'):
    '''f means it's the normal param count branch'''
    b = Conv1D(32, kernel_size=(128), padding='valid', activation='selu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(2), padding='valid', activation='selu')(c) #gives me 128x1
    g = BatchNormalization()(d)
    h = Dense(1,activation='selu')(g)
    return h
def conv_block_double_param_count(input_tensor,conv_activation = 'selu', dense_activation = 'elu'):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    b = Conv1D(64, kernel_size=(128), padding='valid', activation='selu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(2), padding='valid', activation='selu')(c) #gives me 128x1
    g = BatchNormalization()(d)
    h = Dense(4,activation='selu')(g)
    return h
#-----------------------------------------------------------------------------------------------------------------------

#---------------------NARROW WINDOW-------------------------------------------------------------------------------------
def conv_block_double_param_count_narrow_window(input_tensor,conv_activation = 'relu', dense_activation = 'elu'):
    '''requires generator batch for this column to be increased by 28. (15-1) + 2 * (8-1) = 28'''
    b = Conv1D(128, kernel_size=(32), padding='same', activation='relu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(16), padding='same', activation='relu')(c)
    e = BatchNormalization()(d)
    f = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(e)
    g = BatchNormalization()(f)
    h = Dense(4,activation='relu')(g)
    return h
def conv_block_normal_param_count_narrow_window(input_tensor,conv_activation = 'relu', dense_activation = 'elu'):
    '''requires generator batch for this column to be increased by 14. 2 * (8-1) = 14. '''
    b = Conv1D(64, kernel_size=(32), padding='same', activation='relu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(c)
    g = BatchNormalization()(d)
    h = Dense(1,activation='relu')(g)
    return h
#-----------------------------------------------------------------------------------------------------------------------

#---------------------NARROW WINDOW AND CAUSAL--------------------------------------------------------------------------
def conv_block_normal_param_count_narrow_window_causal(input_tensor,conv_activation = 'relu', dense_activation = 'elu'):
    '''requires generator batch for this column to be increased by 14. 2 * (8-1) = 14. '''
    # if you want to use causal: you REALLY need shape[1] (# of steps) to be explicit.
    b = Conv1D(64, kernel_size=(8), padding='causal', activation='relu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(8), padding='causal', activation='relu')(c)
    e = BatchNormalization()(d)
    #f = UpSampling1D(size=2)(e)
    g = BatchNormalization()(e)
    h = Dense(1,activation='relu')(g)
    return h
def conv_block_double_param_count_narrow_window_causal(input_tensor,conv_activation = 'relu', dense_activation = 'elu'):
    '''requires generator batch for this column to be increased by 28. (15-1) + 2 * (8-1) = 28'''
    # if you want to use causal: you REALLY need shape[1] (# of steps) to be explicit.
    b = Conv1D(128, kernel_size=(15), padding='causal', activation='relu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(8), padding='causal', activation='relu')(c)
    e = Conv1D(32, kernel_size=(8), padding='causal', activation='relu')(d)
    f = BatchNormalization()(e)
    #f = UpSampling1D(size=2)(e)
    g = BatchNormalization()(f)
    h = Dense(4,activation='relu')(g)
    return h


def pair_generator_1dconv_lstm(data, labels, start_at=0, generator_batch_size=64, scaled=True, scaler_type ='minmax', use_precomputed_coeffs = True): #shape is something like 1, 11520, 11
    '''Custom batch-yielding generator for Scattergro Output. You need to feed it the numpy array after running "Parse_Individual_Arrays script
    data and labels are self-explanatory.
    Parameters:
        start_at: configures where in the arrays do the generator start yielding (to ensure an LSTM doesn't always start at the same place
        generator_batch_size: how many "rows" of the numpy array does the generator yield each time
        scaled: whether the output is scaled or not.
        scaler_type: which sklearn scaler to call
        scale_what = either the data/label (the whole array), or the yield.'''
    if scaled == True:
        if scaler_type == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
            scaler_step_index_only = sklearn.preprocessing.StandardScaler()
            label_scaler = sklearn.preprocessing.StandardScaler()
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
            label_scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
            label_scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        #print("scaled: {}, scaler_type: {}".format(scaled,scaler_type))

    if use_precomputed_coeffs == True:
        #lists as dummy variables first, seems like scikit flips when I pass in a list as an object attribute..
        scaler_var = [0.6925742052047087, 0.016133766659421164,
                       0.6923827778657753, 0.019533317182529104, 3.621591547512037, 0.03208850741829512,
                       3.621824029181443, 0.03209691975648252, 43.47286356045491, 43.472882235044786]
        scaler_mean = [8.648004880708694, 0.5050077150656151,
                        8.648146575144597, 1.2382993509098987, 9.737983474596277, 1.7792042443537548,
                        9.737976755677462, 1.9832900698119342, 7.859076582026868, 7.859102808059667]
        scaler_scale = [0.8322104332467292, 0.12701876498935566,
                         0.8320954139194466, 0.1397616441751066, 1.9030479624833518, 0.1791326531325183,
                         1.9031090429035966, 0.1791561323440605, 6.593395450028377, 6.5933968661870175]
        label_scaler.var_ = [1.1455965013546072e-11, 1.1571155303166357e-11, 4.3949048693992676e-11, 4.3967045763969097e-11]
        label_scaler.mean_ = [4.5771139469142714e-06, 4.9590312890501306e-06, 6.916592701282579e-06, 6.9171280743598655e-06]
        label_scaler.scale_ = [3.3846661598370483e-06, 3.4016400901868433e-06, 6.6294078690327e-06, 6.63076509642508e-06]
        step_index_to_fit = np.reshape(data[:,0],newshape=(-1,1))
        #print("the shape scikit is bitching about: {}, and after reshape: {}".format(data[:,0].shape, step_index_to_fit.shape))
        scaler_step_index_only.fit(X=step_index_to_fit,y=None) #gotta fit transform since \
        #TODO: /usr/local/lib/python2.7/dist-packages/sklearn/preprocessing/data.py:586: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
        # it makes no sense to precomp the stepindex. reshape is because sklearn gives a warning about 1D arrays as data..
        scaler_var.insert(0,scaler_step_index_only.var_) #append the fitted stepindex params into the main scaler object instance's params.
        scaler.var_ = np.asarray(scaler_var,dtype='float32') #cast as numpy array so scikit won't flip
        scaler_mean.insert(0, scaler_step_index_only.mean_) #set as scaler object attribute
        scaler.mean_ = np.asarray(scaler_mean, dtype='float32')
        scaler_scale.insert(0, scaler_step_index_only.scale_)
        scaler.scale_ = np.asarray(scaler_scale, dtype='float32')
        #print("data scaler mean shape: {} var shape: {} scale shape: {}".format(len(scaler.mean_),len(scaler.var_),len(scaler.scale_)))
        data_scaled = scaler.transform(X=data)
        labels_scaled = label_scaler.transform(X=labels)
    if use_precomputed_coeffs == False:
        data_scaled = scaler.fit_transform(X=data)
        labels_scaled = label_scaler.fit_transform(X=labels)
        #--------i think expand dims is a lot less implicit, that's why i commented these out-------
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))
        #----------------------------------------------------------------------------------------------
        #print("before expand dims: data shape: {}, label shape: {}".format(data_scaled.shape,labels_scaled.shape))

    data_scaled = np.expand_dims(data_scaled, axis=0)  # add 1 dimension in the 0th axis.
    labels_scaled = np.expand_dims(labels_scaled, axis=0)
    index = start_at
    while 1: #for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        #print((data_scaled[:, index:index + generator_batch_size_valid_x1, 0]).shape)
        x1 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x1, 0]),newshape = (1,(generator_batch_size_valid_x1),1)) # first dim = 0 doesn't work.
        x2 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x2, 1]),newshape = (1,(generator_batch_size_valid_x2),1))
        x3 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x3, 2]),newshape = (1,(generator_batch_size_valid_x3),1))
        x4 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x4, 3]),newshape = (1,(generator_batch_size_valid_x4),1))
        x5 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x5, 4]),newshape = (1,(generator_batch_size_valid_x5),1))
        x6 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x6, 5]),newshape = (1,(generator_batch_size_valid_x6),1))
        x7 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x7, 6]),newshape = (1,(generator_batch_size_valid_x7),1))
        x8 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x8, 7]),newshape = (1,(generator_batch_size_valid_x8),1))
        x9 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x9, 8]),newshape = (1,(generator_batch_size_valid_x9),1))
        x10 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x10, 9]),newshape = (1,(generator_batch_size_valid_x10),1))
        x11 = np.reshape((data_scaled[:, index:index + generator_batch_size_valid_x11, 10]),newshape = (1,(generator_batch_size_valid_x11),1))
        y = (labels_scaled[:, index:index + generator_batch_size, :])
        #if generator won't yield the full batch in 3 iterations, then..
        if index + 3 * generator_batch_size < data_scaled.shape[1]:
            index = index + generator_batch_size
        else: #reset. anywhere between 0 and length of dataset - 2*batch size.
            index = np.random.randint(low=0, high=(
            generator_batch_size * ((data_scaled.shape[1] - start_at) // generator_batch_size - 2)))
            # ----------------ENABLE THIS FOR DIAGNOSTICS----------------------
            # print("x_shape at reset: {}".format(x.shape))
        #print("after expand dims:: data shape: {}, x1 shape: {}, x type: {}, y type:{}".format(data_scaled.shape,x1.shape,type(x1),type(y)))
        # x = np.reshape(x,(1,x.shape[0],x.shape[1]))
        # y = np.reshape(y, (1, y.shape[0],y.shape[1]))
        #print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x1.shape, y.shape))
        # if (index == data_scaled.shape[1] - 512): print("index reached: {}".format(index))
        # print("x: {}, y: {}".format(x,y))
        # -------------------ENABLE THIS FOR DIAGNOSTICS-----------------------
        # print("index: {}".format(index))
        # if (x.shape[1] != generator_batch_size_valid and y.shape[1] != generator_batch_size_valid): return
        # if (x.shape[1] != generator_batch_size_valid and y.shape[1] != generator_batch_size_valid): raise StopIteration
        assert (x1.shape[1] == generator_batch_size_valid_x1) #if it's not yielding properly, stop.
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
        yield ([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], y)

#!!!!!!!!!!!!!!!!!!!! TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK THESE FLAGS YO!!!!!!!!!!!!
#shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 3 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 700 #how many times the training corpus is sampled.
generator_batch_size = 128
generator_batch_size_valid_x1 = (generator_batch_size + 128)#4layer conv
generator_batch_size_valid_x2 = (generator_batch_size + 128)
generator_batch_size_valid_x3 = (generator_batch_size + 128)#4layer conv
generator_batch_size_valid_x4 = (generator_batch_size + 128)
generator_batch_size_valid_x5 = (generator_batch_size + 128)#4layer conv
generator_batch_size_valid_x6 = (generator_batch_size + 128)
generator_batch_size_valid_x7 = (generator_batch_size + 128)#4layer conv
generator_batch_size_valid_x8 = (generator_batch_size + 128)
generator_batch_size_valid_x9 = (generator_batch_size + 128)
generator_batch_size_valid_x10 = (generator_batch_size + 128)
generator_batch_size_valid_x11 = (generator_batch_size + 128)
finetune = False
test_only = False #no training. if finetune is also on, this'll raise an error.
use_precomp_sscaler = False
sequence_circumnavigation_amt = 0.5
save_preds = False
save_figs = False
env = "blockade_runner" # "cruiser" "chan" #TODO complete the environment_variable_setter
# generator_batch_size_valid_x1 = (generator_batch_size+28)#4layer conv
# generator_batch_size_valid_x2 = (generator_batch_size+14)
# generator_batch_size_valid_x3 = (generator_batch_size+28)#4layer conv
# generator_batch_size_valid_x4 = (generator_batch_size+14)
# generator_batch_size_valid_x5 = (generator_batch_size+28)#4layer conv
# generator_batch_size_valid_x6 = (generator_batch_size+14)
# generator_batch_size_valid_x7 = (generator_batch_size+28)#4layer conv
# generator_batch_size_valid_x8 = (generator_batch_size+14)
# generator_batch_size_valid_x9 = (generator_batch_size+14)
# generator_batch_size_valid_x10 = (generator_batch_size+14)
# generator_batch_size_valid_x11 = (generator_batch_size+14)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# identifier = "_convlstm_run1_" + str(generator_batch_size) + "b_completev1data_valid_4layer_1357_"
#Weights_200_conv1d_samepad_1_128shortrun

#"_conv1d_samepad_" + str(num_epochs) + "_" + str(generator_batch_size) + "shortrun"
#Weights_200_conv1d_samepad_1_128shortrun
identifier_post_training = "_conv1d_biglstm_ww_128bs_minmaxscaler_adaptive"
#identifier_pre_training = "_conv1d_samepad_" + str(num_epochs) + "_" + str(generator_batch_size) + "shortrun"
identifier_pre_training = "240600_conv1d_selu_widewindow"  #for now, make hardcode what you want to finetune
#@@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis."
#^^^^^^^^^^^^^ TO RUN ON CHEZ CHAN ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Base_Path = "/home/devin/Documents/PITTA LID/"
# image_path = "/home/devin/Documents/PITTA LID/img/"
# train_path = "/home/devin/Documents/PITTA LID/Train FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/Test FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/FV1b 1d nonlinear/"
#+++++++++++++ TO RUN ON LOCAL (IHSAN) +++++++++++++++++++++++++++++++
# Base_Path = "/home/ihsan/Documents/thesis_models/"
# image_path = "/home/ihsan/Documents/thesis_models/images"
# train_path = "/home/ihsan/Documents/thesis_models/train/"
# test_path = "/home/ihsan/Documents/thesis_models/test/"
# analysis_path = "home/ihsan/Documents/thesis_models/analysis"
#%%%%%%%%%%%%% TO RUN ON LOCAL (EFI) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Base_Path = "/home/efi/Documents/thesis_models/"
# image_path = "/home/efi/Documents/thesis_models/images"
# train_path = "/home/efi/Documents/thesis_models/train/"
# test_path = "/home/efi/Documents/thesis_models/test/"
# analysis_path = "home/efi/Documents/thesis_models/analysis"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

g1 = conv_block_double_param_count(input_tensor=a1)
f2 = conv_block_normal_param_count(input_tensor=a2)
g3 = conv_block_double_param_count(input_tensor=a3)
f4 = conv_block_normal_param_count(input_tensor=a4)
g5 = conv_block_double_param_count(input_tensor=a5)
f6 = conv_block_normal_param_count(input_tensor=a6)
g7 = conv_block_double_param_count(input_tensor=a7)
f8 = conv_block_normal_param_count(input_tensor=a8)
f9 = conv_block_normal_param_count(input_tensor=a9)
f10 = conv_block_normal_param_count(input_tensor=a10)
f11 = conv_block_normal_param_count(input_tensor=a11)

#if you want to use causal: you REALLY need shape[1] (# of steps) to be explicit.
# g1 = conv_block_double_param_count_narrow_window_causal(input_tensor=a1)
# f2 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a2)
# g3 = conv_block_double_param_count_narrow_window_causal(input_tensor=a3)
# f4 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a4)
# g5 = conv_block_double_param_count_narrow_window_causal(input_tensor=a5)
# f6 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a6)
# g7 = conv_block_double_param_count_narrow_window_causal(input_tensor=a7)
# f8 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a8)
# f9 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a9)
# f10 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a10)
# f11 = conv_block_normal_param_count_narrow_window_causal(input_tensor=a11)

# define the model first

tensors_to_concat = [g1, f2, g3, f4, g5, f6, g7, f8, f9, f10, f11]
g = concatenate(tensors_to_concat)
out = reference_bilstm(input_tensor=g)


model = Model(inputs=[a1,a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], outputs=out)
plot_model(model, to_file='model_' + identifier_pre_training + '.png',show_shapes=True)
optimizer_used = adam(lr=0.002)
model.compile(loss='mse', optimizer=optimizer_used, metrics=['accuracy', 'mae', 'mape', 'mse'])
print("Model summary: {}".format(model.summary()))

print("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Metrics: {}".format(model.metrics_names))

#load data multiple times.
data_filenames = os.listdir(train_path + "data")
#print("before sorting, data_filenames: {}".format(data_filenames))
data_filenames.sort()
#print("after sorting, data_filenames: {}".format(data_filenames))

label_filenames = os.listdir(train_path + "label")
label_filenames.sort() #sorting makes sure the label and the data are lined up.
#print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames,label_filenames)
#print("before shuffling: {}".format(combined_filenames))
shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.
print('Data loaded.')
weights_file_name = None

if finetune==False:
    weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
    print("Are weights (with the given name to be saved as) already present? {}".format(weights_present_indicator))
else:
    weights_present_indicator = os.path.isfile('Weights_' + identifier_pre_training + '.h5')
    print("Are weights (with the given name) to initialize with present? {}".format(weights_present_indicator))

csv_logger = CSVLogger(filename = './analysis/logtrain' + identifier_post_training + ".csv", append=True)

if (finetune == False and weights_present_indicator == False and test_only == False) or (finetune == True and weights_present_indicator == True):
    print("TRAINING PHASE")
    print("weights_present_indicator: {}, finetune: {}".format(weights_present_indicator, finetune))
    for i in range(0, num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        print("files: {}".format(files))
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape, label_array.shape, i))
        # train_array = np.reshape(train_array,(1,generator_batch_size,train_array.shape[1]))
        # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!


        nonlinear_part_starting_position = generator_batch_size * ((train_array.shape[0] // generator_batch_size) - 5)
        shuffled_starting_position = np.random.randint(0, nonlinear_part_starting_position)

        if finetune == True:  # load the weights
            finetune_init_weights_filename = 'Weights_' + identifier_pre_training + '.h5'  # hardcode the previous epoch number UP ABOVE
            model.load_weights(finetune_init_weights_filename, by_name=True)

        train_generator = pair_generator_1dconv_lstm(train_array, label_array, start_at=shuffled_starting_position,
                                                     generator_batch_size=generator_batch_size,
                                                     use_precomputed_coeffs=use_precomp_sscaler)
        training_hist = model.fit_generator(train_generator, epochs=num_epochs,
                                            steps_per_epoch=sequence_circumnavigation_amt * (
                                            train_array.shape[0] // generator_batch_size), verbose=2,
                                            callbacks=[csv_logger])

    if weights_present_indicator == True and finetune == True:
        print("fine-tuning/partial training session completed.")
        weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
        model.save_weights(weights_file_name)
        print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                         weights_file_name))
    if weights_present_indicator == False and finetune == False:  # fresh training
        print("FRESH training session completed.")
        weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
        model.save_weights(weights_file_name)
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
    weights_file_name = weights_to_test_with_fname #piggybacking the old flag. the one without fname is to refer to post training weights.
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
        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at=0,
                                                    generator_batch_size=generator_batch_size,
                                                    use_precomputed_coeffs=use_precomp_sscaler)
        for i in range(1):
            X_test_batch, y_test_batch = test_generator.next()
            # print(X_test_batch)
            # print(y_test_batch)
            score = model.predict_on_batch(X_test_batch)
            # print("Score: {}".format(score)) #test_array.shape[1]//generator_batch_size
        score = model.evaluate_generator(test_generator, steps=(test_array.shape[0] // generator_batch_size),
                                         max_queue_size=test_array.shape[0], use_multiprocessing=False)
        row_dict = {}
        print("scores: {}".format(score))
        row_dict['filename'] = str(files[0])[:-4]
        row_dict['loss'] = score[0]  # 'loss'
        row_dict['acc'] = score[1]  # 'acc'
        row_dict['mae'] = score[2]  # 'mean_absolute_error'
        row_dict['mape'] = score[3]  # 'mean_absolute_percentage_error'
        score_rows_list.append(row_dict)

        # testing should start at 0. For now.
        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at=0,
                                                    generator_batch_size=generator_batch_size,
                                                    use_precomputed_coeffs=use_precomp_sscaler)
        prediction_length = (int(1.0 * (generator_batch_size * (label_array.shape[0] // generator_batch_size))))
        test_i = 0
        # Kindly declare the shape
        x_prediction = np.zeros(shape=[1, prediction_length, 4])
        y_prediction = np.zeros(shape=[1, prediction_length, 4])

        while test_i <= prediction_length - generator_batch_size:
            x_test_batch, y_test_batch = test_generator.next()
            x_prediction[0, test_i:test_i + generator_batch_size, :] = model.predict_on_batch(x_test_batch)
            y_prediction[0, test_i:test_i + generator_batch_size, :] = y_test_batch
            test_i += generator_batch_size
        # print("array shape {}".format(y_prediction[0,int(0.95*prediction_length), :].shape))
        if save_preds == True:
            np.save(Base_Path + 'predictionbatch' + str(files[0]), arr=y_prediction)

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
    score_df.to_csv('scores_lstm_' + identifier_post_training + '.csv')
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