from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, Dense, Dropout, \
    Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, concatenate
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing

def np_array_pair_generator(data,labels,start_at=0,generator_batch_size=64,scaled=True,scaler_type = 'standard',scale_what = 'data'): #shape is something like 1, 11520, 11
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
            #print('standard scaler initialized: {}'.format(scaler))
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        #print("scaled: {}, scaler_type: {}".format(scaled,scaler_type))
        data_scaled = scaler.fit_transform(X=data, y=None)
        labels_scaled = scaler.fit_transform(X=labels, y=None) #i don't think you should scale the labels..
        #labels_scaled = labels #don't scale the labels..
        #--------i think expand dims is a lot less implicit, that's why i commented these out-------
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))
        #----------------------------------------------------------------------------------------------
        #print("before expand dims: data shape: {}, label shape: {}".format(data_scaled.shape,labels_scaled.shape))
        data_scaled = np.expand_dims(data_scaled, axis=0)  # add 1 dimension in the
        labels_scaled = np.expand_dims(labels_scaled, axis=0)
        index = start_at
    while 1: #for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        x1 = np.reshape((data_scaled[:, index:index + generator_batch_size, 0]),newshape = (1,generator_batch_size,1)) # first dim = 0 doesn't work.
        x2 = np.reshape((data_scaled[:, index:index + generator_batch_size, 1]),newshape = (1,generator_batch_size,1))
        x3 = np.reshape((data_scaled[:, index:index + generator_batch_size, 2]),newshape = (1,generator_batch_size,1))
        x4 = np.reshape((data_scaled[:, index:index + generator_batch_size, 3]),newshape = (1,generator_batch_size,1))
        x5 = np.reshape((data_scaled[:, index:index + generator_batch_size, 4]),newshape = (1,generator_batch_size,1))
        x6 = np.reshape((data_scaled[:, index:index + generator_batch_size, 5]),newshape = (1,generator_batch_size,1))
        x7 = np.reshape((data_scaled[:, index:index + generator_batch_size, 6]),newshape = (1,generator_batch_size,1))
        x8 = np.reshape((data_scaled[:, index:index + generator_batch_size, 7]),newshape = (1,generator_batch_size,1))
        x9 = np.reshape((data_scaled[:, index:index + generator_batch_size, 8]),newshape = (1,generator_batch_size,1))
        x10 = np.reshape((data_scaled[:, index:index + generator_batch_size, 9]),newshape = (1,generator_batch_size,1))
        x11 = np.reshape((data_scaled[:, index:index + generator_batch_size, 10]),newshape = (1,generator_batch_size,1))
        y = (labels_scaled[:, index:index + generator_batch_size, :])
        #if generator won't yield the full batch in 3 iterations, then..
        if index + 3 * generator_batch_size < data_scaled.shape[1]:
            index = index + generator_batch_size
        else: #reset. anywhere between 0 and length of dataset - 3*batch size.
            index = np.random.randint(low=0, high=(
            generator_batch_size * ((data_scaled.shape[1] - start_at) // generator_batch_size - 3)))
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
        # if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): return
        # if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): raise StopIteration
        assert (x1.shape[1] == generator_batch_size) #if it's not yielding properly, stop.
        # assert(y.shape[1]==generator_batch_size)
        yield ([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], y)


#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 5 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 500 #how many times the training corpus is sampled.
generator_batch_size = 1024
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_conv1d_run3_1024b_completev1data_"
Base_Path = "/home/ihsan/Documents/thesis_models/"
#train_path = "/home/devin/Documents/PITTA LID/train/"
#test_path = "/home/devin/Documents/PITTA LID/test/"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#11 input columns
#4 output columns.

np.random.seed(1337)
# define the model first
a1 = Input(shape=(None, 1))
b1 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a1)
#c1 = MaxPooling1D((2))(b1)
d1 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b1)
#e1 = Flatten()(d1)
f1 = Dense(1, activation='selu')(d1)

a2 = Input(shape=(None, 1))
b2 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a2)
#c2 = MaxPooling1D((2))(b2)
d2 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b2)
#e2 = Flatten()(d2)
f2 = Dense(8, activation='selu')(d2)

a3 = Input(shape=(None, 1))
b3 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a3)
#c3 = MaxPooling1D((2))(b3)
d3 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b3)
#e3 = Flatten()(d3)
f3 = Dense(8, activation='selu')(d3)

a4 = Input(shape=(None, 1))
b4 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a4)
#c4 = MaxPooling1D((2))(b4)
d4 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b4)
#e4 = Flatten()(d4)
f4 = Dense(8, activation='selu')(d4)

a5 = Input(shape=(None, 1))
b5 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a5)
#c5 = MaxPooling1D((2))(b5)
d5 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b5)
#e5 = Flatten()(d5)
f5 = Dense(8, activation='selu')(d5)

a6 = Input(shape=(None, 1))
b6 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a6)
#c6 = MaxPooling1D((2))(b6)
d6 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b6)
#e6 = Flatten()(d6)
f6 = Dense(8, activation='selu')(d6)

a7 = Input(shape=(None, 1))
b7 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a7)
#c7 = MaxPooling1D((2))(b7)
d7 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b7)
#e7 = Flatten()(d7)
f7 = Dense(8, activation='selu')(d7)

a8 = Input(shape=(None, 1))
b8 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a8)
#c8 = MaxPooling1D((2))(b8)
d8 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b8)
#e8 = Flatten()(d8)
f8 = Dense(8, activation='selu')(d8)

a9 = Input(shape=(None, 1))
b9 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a9)
#c9 = MaxPooling1D((2))(b9)
d9 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b9)
#e9 = Flatten()(d9)
f9 = Dense(8, activation='selu')(d9)

a10 = Input(shape=(None, 1))
b10 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a10)
#c10 = MaxPooling1D((2))(b10)
d10 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b10)
#e10 = Flatten()(d10)
f10 = Dense(1, activation='selu')(d10)

a11 = Input(shape=(None, 1))
b11 = Conv1D(32, kernel_size=(8), padding='same', activation='relu')(a11)
#c11 = MaxPooling1D((2))(b11)
d11 = Conv1D(16, kernel_size=(8), padding='same', activation='relu')(b11)
#e11 = Flatten()(d11)
f11 = Dense(1, activation='selu')(d11)

g = concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])
h = Dense(16,activation='selu')(g)
#h = Flatten()(g)
out = Dense(4)(h)

model = Model(inputs=[a1,a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], outputs=out)
plot_model(model, to_file='model_' + identifier + '.png',show_shapes=False)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mae', 'mape', 'mse'])
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
        #train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        train_generator = np_array_pair_generator(train_array,label_array,start_at=0,generator_batch_size=generator_batch_size)

        train_generator = np_array_pair_generator(train_array,label_array,start_at=0,generator_batch_size=generator_batch_size)
        training_hist = model.fit_generator(train_generator,epochs=num_epochs,steps_per_epoch=3*(train_array.shape[0]//generator_batch_size),verbose=2)

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
    #TODO: still only saves single results.
    for files in combined_filenames:
        i=i+1
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        print("file: {} data/label shape: {}, {}".format(files[0],test_array.shape, label_array.shape))
        print("Metrics: {}".format(model.metrics_names))
        # steps per epoch is how many times that generator is called
        test_generator = np_array_pair_generator(test_array, label_array, start_at = 0,generator_batch_size=generator_batch_size)
        # for i in range (batch_size):
        #     X_test_batch, y_test_batch = test_generator.next()
        #     score = model.predict_on_batch(X_test_batch,y_test_batch)
        #     print("Score: {}".format(score)) #test_array.shape[1]//generator_batch_size
        score = model.evaluate_generator(test_generator, steps=(test_array.shape[0]//generator_batch_size),max_queue_size=test_array.shape[0],use_multiprocessing=False)
        print("scores: {}".format(score))
        #home/ihsan/Documents/thesis_models/results/
        np.savetxt('TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
                   fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
                   footer=str(), comments='# ')
