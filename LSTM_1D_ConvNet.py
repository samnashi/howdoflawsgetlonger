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
import pandas as pd
import scipy.io as sio
from keras.callbacks import CSVLogger
import os
import csv
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing

# np.set_printoptions(threshold='nan')

def conv_block_normal_param_count(input_tensor,i=0):
    '''f means it's the normal param count branch'''
    b = Conv1D(64, kernel_size=(64), padding='valid', activation='elu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(32, kernel_size=(16), padding='valid', activation='elu')(c)
    e = BatchNormalization()(d)
    f = Dense(1, activation='elu')(e)
    return f

def conv_block_double_param_count(input_tensor,i=0):
    '''g means it's the output of the "twice the number of parameters"  branch'''
    b = Conv1D(128, kernel_size=(128), padding='valid', activation='elu')(input_tensor)
    c = BatchNormalization()(b)
    d = Conv1D(64, kernel_size=(16), padding='valid', activation='elu')(c)
    e = BatchNormalization()(d)
    e = Conv1D(64, kernel_size=(4), padding='valid', activation='elu')(d)
    f = BatchNormalization()(e)
    g = Dense(4, activation='elu')(f)
    return g

def pair_generator_1dconv_lstm(data, labels, start_at=0, generator_batch_size=64, scaled=True, scaler_type ='standard', use_precomputed_coeffs = True): #shape is something like 1, 11520, 11
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
            label_scaler = sklearn.preprocessing.StandardScaler()
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        #print("scaled: {}, scaler_type: {}".format(scaled,scaler_type))

    if use_precomputed_coeffs == True:
        scaler.var_ = [1283.8767902599698, 0.6925742052047087, 0.016133766659421164,
                       0.6923827778657753, 0.019533317182529104, 3.621591547512037, 0.03208850741829512,
                       3.621824029181443, 0.03209691975648252, 43.47286356045491, 43.472882235044786]
        scaler.mean_ = [77.84198603763824, 8.648004880708694, 0.5050077150656151,
                        8.648146575144597, 1.2382993509098987, 9.737983474596277, 1.7792042443537548,
                        9.737976755677462, 1.9832900698119342, 7.859076582026868, 7.859102808059667]
        scaler.scale_ = [35.831226468821434, 0.8322104332467292, 0.12701876498935566, 0.8320954139194466, 0.1397616441751066, 1.9030479624833518, 0.1791326531325183, 1.9031090429035966, 0.1791561323440605, 6.593395450028377, 6.5933968661870175]
        label_scaler.var_ = [1.1455965013546072e-11, 1.1571155303166357e-11, 4.3949048693992676e-11, 4.3967045763969097e-11]
        label_scaler.mean_ = [4.5771139469142714e-06, 4.9590312890501306e-06, 6.916592701282579e-06, 6.9171280743598655e-06]
        label_scaler.scale_ = [3.3846661598370483e-06, 3.4016400901868433e-06, 6.6294078690327e-06, 6.63076509642508e-06]
        data_scaled = scaler.transform(X=data,y=None)
        labels_scaled = label_scaler.transform(X=labels,y=None)
    if use_precomputed_coeffs == False:
        data_scaled = scaler.fit_transform(X=data, y=None)
        labels_scaled = scaler.fit_transform(X=labels, y=None)
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


#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 5 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 5 #how many times the training corpus is sampled.
generator_batch_size = 256
generator_batch_size_valid_x1 = (generator_batch_size)#4layer conv
generator_batch_size_valid_x2 = (generator_batch_size)
generator_batch_size_valid_x3 = (generator_batch_size)#4layer conv
generator_batch_size_valid_x4 = (generator_batch_size)
generator_batch_size_valid_x5 = (generator_batch_size)#4layer conv
generator_batch_size_valid_x6 = (generator_batch_size)
generator_batch_size_valid_x7 = (generator_batch_size)#4layer conv
generator_batch_size_valid_x8 = (generator_batch_size)
generator_batch_size_valid_x9 = (generator_batch_size)
generator_batch_size_valid_x10 = (generator_batch_size)
generator_batch_size_valid_x11 = (generator_batch_size)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# identifier = "_convlstm_run1_" + str(generator_batch_size) + "b_completev1data_valid_4layer_1357_"
identifier = "_conv1d_run2_" + str(generator_batch_size) + "ihsanconfig"
# Base_Path = "/home/devin/Documents/PITTA LID/"
# image_path = "/home/devin/Documents/PITTA LID/img/"
# train_path = "/home/devin/Documents/PITTA LID/Train FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/Test FV1b/"
# test_path = "/home/devin/Documents/PITTA LID/FV1b 1d nonlinear/"
#+++++++++++++++TO RUN ON LOCAL (IHSAN)+++++++++++++++++++++++++++++++
Base_Path = "/home/ihsan/Documents/thesis_models/"
image_path = "/home/ihsan/Documents/thesis_models/images"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#11 input columns
#4 output columns.

#TODO ADD AND CHECK BATCH NORMALIZATION!!
#TODO make this convnet construction be a loop.. this is a nightmare to keep track of
np.random.seed(1337)
# define the model first

#keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, \
                                  #activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#TODO: make this dependent on the batch?
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

# b1 = Conv1D(128, kernel_size=(64), strides=1, dilation_rate=1, padding='valid', activation='relu')(a1)
# #print("b1's type: {} shape: {}".format(type(b1), b1.shape))
# c1 = BatchNormalization()(b1)
# d1 = Conv1D(64, kernel_size=(8), strides=1, dilation_rate=1, padding='valid', activation='relu')(b1)
# e1 = BatchNormalization()(d1)
# # e1 = Conv1D(8, kernel_size=(8),  strides=1, dilation_rate=1, padding='valid', activation='relu')(d1)
# # f1 = Conv1D(8, kernel_size=(8),  strides=1, dilation_rate=1, padding='valid', activation='relu')(e1)
# g1 = Dense(1, activation='elu')(d1)
# # f1 = Dense(1, activation='selu')(e1)
#
#
#
# b2 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a2)
# #c2 = MaxPooling1D((2))(b2)
# d2 = Conv1D(16, kernel_size=(8), padding='valid', activation='relu')(b2)
# # e2 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d2)
# e2 = Flatten()(d2)
# #f2 = Dense(8, activation='elu')(d2)
# # f2 = Dense(8, activation='selu')(e2)
#
#
# b3 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a3)
# #c3 = MaxPooling1D((2))(b3)
# d3 = Conv1D(16, kernel_size=(8), padding='valid', activation='relu')(b3)
# e3 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(d3)
# f3 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(e3)
# e3 = Flatten()(d3)
# #g3 = Dense(8, activation='elu')(f3)
# # f3 = Dense(8, activation='selu')(e3)
#
#
# b4 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a4)
# #c4 = MaxPooling1D((2))(b4)
# d4 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b4)
# # e4 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d4)
# e4 = Flatten()(d4)
# f4 = Dense(8, activation='elu')(d4)
#
#
# b5 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a5)
# #c5 = MaxPooling1D((2))(b5)
# d5 = Conv1D(16, kernel_size=(8), padding='valid', activation='relu')(b5)
# e5 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(d5)
# f5 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(e5)
# #e5 = Flatten()(d5)
# g5 = Dense(8, activation='elu')(f5)
#
#
# b6 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a6)
# #c6 = MaxPooling1D((2))(b6)
# d6 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b6)
# # e6 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d6)
# #e6 = Flatten()(d6)
# f6 = Dense(8, activation='elu')(d6)
#
#
# b7 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a7)
# #c7 = MaxPooling1D((2))(b7)
# d7 = Conv1D(16, kernel_size=(8), padding='valid', activation='relu')(b7)
# e7 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(d7)
# f7 = Conv1D(8, kernel_size=(8), padding='valid', activation='relu')(e7)
# #e7 = Flatten()(d7)
# g7 = Dense(8, activation='elu')(f7)
#
#
# b8 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a8)
# #c8 = MaxPooling1D((2))(b8)
# d8 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b8)
# # e8 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d8)
# #e8 = Flatten()(d8)
# f8 = Dense(8, activation='elu')(d8)
#
#
# b9 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a9)
# #c9 = MaxPooling1D((2))(b9)
# d9 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b9)
# # e9 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d9)
# #e9 = Flatten()(d9)
# f9 = Dense(8, activation='elu')(d9)
#
#
# b10 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a10)
# #c10 = MaxPooling1D((2))(b10)
# d10 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b10)
# # e10 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d10)
# #e10 = Flatten()(d10)
# f10 = Dense(1, activation='elu')(d10)
#
#
# b11 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(a11)
# #c11 = MaxPooling1D((2))(b11)
# d11 = Conv1D(4, kernel_size=(8), padding='valid', activation='relu')(b11)
# # e11 = Conv1D(32, kernel_size=(8), padding='valid', activation='relu')(d11)
# #e11 = Flatten()(d11)
# f11 = Dense(1, activation='elu')(d11)

g = concatenate([g1, f2, g3, f4, g5, f6, g7, f8, f9, f10, f11])
h = Bidirectional(LSTM(128,kernel_initializer=lecun_normal(seed=1337),return_sequences=True))(g)
i = BatchNormalization()(h)
j = Bidirectional(LSTM(128,kernel_initializer=lecun_normal(seed=1337),return_sequences=True))(i)
k = TimeDistributed(Dense(64,kernel_initializer=lecun_normal(seed=1337), activation='elu'))(j)
l = Dense(16,activation='elu')(k)
out = Dense(4)(l)

model = Model(inputs=[a1,a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], outputs=out)
plot_model(model, to_file='model_' + identifier + '.png',show_shapes=True)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mae', 'mape', 'mse'])
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
print('loading data...')

print("weights present? {}".format((os.path.isfile(Base_Path + 'Weights_' + str(num_sequence_draws) + identifier + '.h5'))))
if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    print("TRAINING PHASE")

    for i in range(0,num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        print("files: {}".format(files))
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape,label_array.shape, i))
        # train_array = np.reshape(train_array,(1,generator_batch_size,train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        train_generator = pair_generator_1dconv_lstm(train_array, label_array, start_at=0, generator_batch_size=generator_batch_size, use_precomputed_coeffs=True)
        training_hist = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=3*(train_array.shape[0]//generator_batch_size), verbose=2)

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
    weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier + '.h5'
    print("after {} iterations, model weights is saved as {}".format(num_sequence_draws*num_epochs, weights_file_name))
    model.save_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')

if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == True:
    #the testing part
    print("TESTING PHASE, with weights {}".format('Weights_' + str(num_sequence_draws) + identifier + '.h5'))
    model.load_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')

    # load data multiple times.
    data_filenames = os.listdir(test_path + "data")
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))


    label_filenames = os.listdir(test_path + "label")
    label_filenames.sort()
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_filenames))
    shuffle(combined_filenames)
    print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.

    i=0
    scaler_output = sklearn.preprocessing.StandardScaler()
    #TODO: still only saves single results.
    for files in combined_filenames:
        csv_logger = CSVLogger('logtest.log')
        i=i+1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        # print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
        print(files[0])
        # print("Metrics: {}".format(model.metrics_names))
        # steps per epoch is how many times that generator is called
        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at = 0, generator_batch_size=generator_batch_size, use_precomputed_coeffs=True)
        for i in range (1):
            X_test_batch, y_test_batch = test_generator.next()
            # print(X_test_batch)
            # print(y_test_batch)
            score = model.predict_on_batch(X_test_batch)
            # print("Score: {}".format(score)) #test_array.shape[1]//generator_batch_size
        score = model.evaluate_generator(test_generator, steps=(test_array.shape[0]//generator_batch_size),max_queue_size=test_array.shape[0],use_multiprocessing=False)
        print("scores: {}".format(score))
        # print(score)
        #home/ihsan/Documents/thesis_models/results/
        np.savetxt('TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
                   fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
                   footer=str(), comments='# ')

        test_generator = pair_generator_1dconv_lstm(test_array, label_array, start_at = 0, generator_batch_size=generator_batch_size, use_precomputed_coeffs=True)
        prediction_length = (int(0.85*(generator_batch_size * (label_array.shape[0]//generator_batch_size))))
        test_i=0
        # Kindly declare the shape
        x_prediction = np.zeros(shape=[1, prediction_length, 4])
        y_prediction = np.zeros(shape=[1, prediction_length, 4])


        while test_i <= prediction_length - generator_batch_size:
            x_test_batch, y_test_batch = test_generator.next()
            x_prediction[0,test_i:test_i + generator_batch_size,:] = model.predict_on_batch(x_test_batch)
            y_prediction[0,test_i:test_i + generator_batch_size,:] = y_test_batch
            test_i += generator_batch_size
        # print("array shape {}".format(y_prediction[0,int(0.95*prediction_length), :].shape))
        np.save(Base_Path + 'predictionbatch' + str(files[0]), arr=y_prediction)

        # print(y_prediction.shape)
        # print (x_prediction.shape)
        # print ("label array shape: {}".format(label_array.shape))

        # print("y_prediction shape: {}".format(y_prediction.shape))
        y_prediction_temp = y_prediction
        y_prediction = np.reshape(y_prediction,newshape=(y_prediction_temp.shape[1],y_prediction_temp.shape[2]))
        label_truth = label_array[0:y_prediction.shape[0],:]
        # print (label_truth.shape)
        label_truth_temp = label_truth
        label_truth = scaler_output.fit_transform(X=label_truth_temp,y=None)

        resample_interval = 16
        label_truth = label_truth[::resample_interval,:]
        y_prediction = y_prediction[::resample_interval,:]

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

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,0],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,0],'.',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction from 50% - 100% of the sequence on Crack 01')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,1],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,1],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 02')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,2],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,2],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 03')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')

        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(label_truth[:,3],'^',label="ground truth", markersize=5)
        plt.plot(y_prediction[:,3],'v',label="prediction", markersize=4)
        plt.xscale('log')
        plt.xlabel('# Cycle(s)')
        plt.yscale('log')
        plt.ylabel('Value(s)')
        plt.legend()
        plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
        plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 04')
        plt.grid(True)
        plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')