from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional
from keras.initializers import lecun_normal,glorot_normal
from keras.optimizers import rmsprop
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing


# def batch_size_verifier(generator_batch_size = 64,array_size = 10000,steps_per_epoch=5):
#     limit = generator_batch_size * (array_size//generator_batch_size)
#     minimum_batch_size =

#you limit the # of calls keras calls the generator OUTSIDE the generator.
#each time you fit, dataset length // batch size. round down!
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
            print('Standard Scaler initialized \n')
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler() #TRY NORMALIZER FOR THE LABEL
        #print("scaled: {}, scaler_type: {}".format(scaled, scaler_type))
        data_scaled = scaler.fit_transform(X=data, y=None)
        labels_scaled = scaler.fit_transform(X=labels, y=None) #i don't think you should scale the labels..
        #labels_scaled = labels
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))
        data_scaled = np.expand_dims(data_scaled, axis=0)  # add 1 dimension in the
        labels_scaled = np.expand_dims(labels_scaled, axis=0)
        index = start_at
    while 1:
        #if index < ((data_scaled.shape[1]-start_at)//generator_batch_size)* generator_batch_size:  # for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        #while index < ((data_scaled.shape[1]-start_at)//generator_batch_size)* generator_batch_size: # for index in range(start_at,data_scaled.shape[1]):
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x = (data_scaled[:, index:index + generator_batch_size, :])  # first dim = 0 doesn't work.
        y = (labels_scaled[:, index:index + generator_batch_size, :])  # yield shape = (4,)
        #generator_batch_size * (data_scaled.shape[1] - start_at) // generator_batch_size
        if index + 2 * generator_batch_size < data_scaled.shape[1]:
            index = index + generator_batch_size
        else:
            index = np.random.randint(low=0,high=(generator_batch_size*((data_scaled.shape[1]-start_at)//generator_batch_size-2)))
            #----------------ENABLE THIS FOR DIAGNOSTICS----------------------
            #print("x_shape at reset: {}".format(x.shape))
        # print("data shape: {}, x type: {}, y type:{}".format(data_scaled.shape,type(x),type(y)))
        # x = np.reshape(x,(1,x.shape[0],x.shape[1]))
        # y = np.reshape(y, (1, y.shape[0],y.shape[1]))
        #print("after reshaping: index: {}, x shape: {}, y shape:{}".format(index, x.shape, y.shape))
        #if (index == data_scaled.shape[1] - 512): print("index reached: {}".format(index))
        # print("x: {}, y: {}".format(x,y))
        #-------------------ENABLE THIS FOR DIAGNOSTICS-----------------------
        #print("index: {}".format(index))
        #if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): return
        # if (x.shape[1] != generator_batch_size and y.shape[1] != generator_batch_size): raise StopIteration
        assert(x.shape[1]==generator_batch_size)
        # assert(y.shape[1]==generator_batch_size)
        yield (x, y)

#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 5 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 500 #how many times the training corpus is sampled.
generator_batch_size = 256
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_3c_elu_longtrain_256bat_fv1b_"
Base_Path = "./"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#11 input columns
#4 output columns.

np.random.seed(1337)
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

#define the model first
a = Input(shape=(None,11))
b = Bidirectional(LSTM(200,kernel_initializer=lecun_normal(seed=1337),return_sequences=True))(a)
c = Bidirectional(LSTM(200,kernel_initializer=lecun_normal(seed=1337),return_sequences=True))(b)
d = TimeDistributed(Dense(64,activation='elu',kernel_initializer=lecun_normal(seed=1337)))(c)
out = TimeDistributed(Dense(4,kernel_initializer=lecun_normal(seed=1337)))(d)

keras_optimizer = rmsprop(lr=0.0015, rho=0.9, epsilon=1e-08, decay=0.0)
model = Model(inputs=a,outputs=out)
model.compile(loss='mse', optimizer=keras_optimizer,metrics=['accuracy','mae','mape','mse'])
print("Model summary: {}".format(model.summary()))
print("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Metrics: {}".format(model.metrics_names))

plot_model(model, to_file='model_' + identifier + '.png',show_shapes=True)
#print ("Actual input: {}".format(data.shape))
#print ("Actual output: {}".format(target.shape))

print('loading data.')
weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#HARDCODED
#weights_present_indicator = True
if weights_present_indicator == False:
    print("TRAINING PHASE")

    for i in range(0,num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:,1:]
        #-----------COMMENT THESE OUT IF YOU WANT RESCALER ON----------------------------------
        #train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
        #--------------------------------------------------------------------------------------
        print("filename: {}, data/label shape: {}, {}, draw #: {}".format(str(files[0]),train_array.shape,label_array.shape, i))

        generator_starting_index = train_array.shape[1] - 1 - shortest_length #steps per epoch is how many times that generator is called #train_array.shape[0]//generator_batch_size
        training_hist = model.fit_generator(np_array_pair_generator(train_array,label_array,start_at=0,generator_batch_size=generator_batch_size),epochs=num_epochs,steps_per_epoch=3*(train_array.shape[0]//generator_batch_size),verbose=2)

    #model.save('Model_' + str(num_sequence_draws) + identifier + '.h5')
    if os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5') == False:
        weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier + '.h5'
        model.save_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')
        print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                         weights_file_name))
    print('training_hist keys: {}'.format(training_hist.history.keys()))

    best_epoch = np.argmax(np.asarray(training_hist.history['acc']))

    best_result = np.asarray((best_epoch, (np.asarray(training_hist.history['loss'])[best_epoch]),
                              (np.asarray(training_hist.history['acc'])[best_epoch]),
                              (np.asarray(training_hist.history['mean_absolute_percentage_error'])[best_epoch]),
                              (np.asarray(training_hist.history['mean_absolute_error'])[best_epoch])))
    print('best epoch index: {}, best result: {}'.format(best_epoch,
                                                         best_result))  # actual epoch is index+1 because arrays start at 0..

    # # saves the best epoch's results
    np.savetxt(Base_Path + 'results/BestEpochResult_' + str(num_sequence_draws) + identifier + '.txt', best_result,
               fmt='%5.6f', delimiter=' ', newline='\n', header='epoch, loss, acc, mape, mae',
               footer=str(), comments='# ')

    np.save(Base_Path + 'results/acc_' + str(num_sequence_draws) + identifier + '.npy',
            np.asarray(training_hist.history['acc']))
    np.save(Base_Path + 'results/loss_' + str(num_sequence_draws) + identifier + '.npy',
            np.asarray(training_hist.history['loss']))

    # # summarize history for accuracy
    plt.plot(training_hist.history['acc'])
    plt.title('model MAIN accuracy' + identifier)
    plt.ylabel('MAIN accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig(Base_Path + 'results/main_acc_' + str(num_sequence_draws) + identifier + '.png', bbox_inches='tight')
    plt.clf()

    # # summarize history for loss
    plt.plot(training_hist.history['loss'])
    plt.title('model loss' + identifier)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig(Base_Path + 'results/loss_' + str(num_sequence_draws) + identifier + '.png', bbox_inches='tight')
    plt.clf()

weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#weights_present_indicator = True
if weights_present_indicator == True:
    #the testing part
    print("TESTING PHASE, with weights {}".format('Weights_' + str(num_sequence_draws) + identifier + '.h5'))
    #print("TESTING PHASE, with weights {}".format('Weights_300_3_firstrun_fv1b_server'))
    model.load_weights('Weights_' + str(num_sequence_draws) + identifier + '.h5')
    #model.load_weights('Weights_300_3_firstrun_fv1b_server.h5')

    # load data multiple times.

    #data_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/data/')
    data_filenames = os.listdir(test_path + "data")
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    #label_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/label')
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
    score_rows_list = []
    for files in combined_filenames:
        i=i+1
        # data_load_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/data/' + files[0]
        # label_load_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/label/' + files[1]
        data_load_path = test_path + '/data/' + files[0]
        label_load_path = test_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        test_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
        #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
        #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
        print("filename: {}, data/label shape: {}, {}".format(str(files[0]),test_array.shape, label_array.shape))
        predictions_length = generator_batch_size * (label_array.shape[0]//generator_batch_size)
        #largest integer multiple of the generator batch size that fits into the length of the sequence.

        test_generator = np_array_pair_generator(test_array, label_array, start_at = 0,generator_batch_size=generator_batch_size)
        row_dict = {}
        score = model.evaluate_generator(test_generator, steps=(1 * test_array.shape[0] // generator_batch_size),
                                         max_queue_size=test_array.shape[0], use_multiprocessing=False)
        print("scores: {}".format(score))
        row_dict['filename'] = str(files[0])[:-4]
        row_dict['loss'] = score[0] #'loss'
        row_dict['acc'] = score[1] #'acc'
        row_dict['mae'] =score[2] #'mean_absolute_error'
        row_dict['mape'] = score[3] #'mean_absolute_percentage_error'
        score_rows_list.append(row_dict)

    score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())
    score_df.to_csv('scores_lstm_' + identifier + '.csv')

        # y_pred = np.zeros(shape=(1,predictions_length,4))
        # y_true = np.zeros(shape=(1,predictions_length,4))
        # X_test_batch, y_test_batch = test_generator.next()
        # print("X_test_batch shape: {}, y_test_batch_shape: {}".format(X_test_batch.shape,y_test_batch.shape))
        # test_i = 0
        # while test_i <= predictions_length - generator_batch_size:
        #     #print("test_i: {}".format(test_i))
        #     X_test_batch, y_test_batch = test_generator.next()
        #     y_pred[0,test_i:test_i + generator_batch_size,:] = model.predict_on_batch(X_test_batch)
        #     y_true[0,test_i:test_i + generator_batch_size,:] = y_test_batch
        #     test_i += generator_batch_size
        # print("array to print's shape: {}".format(y_pred[0, int(0.75*predictions_length):, :].shape))
        # #------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------------------------
        # #np.save(file=('predictions_lstm_'+str(files[0])),arr=y_pred)
        # # ------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------------------------
        # resample_interval = 8
        # axis_option = 'symlog'
        # y_pred = y_pred[::resample_interval, :]
        # y_true = y_true[::resample_interval, :]
        # #x_range= np.arange(start=0, stop=y_true.shape[1])
        # # plt.scatter(x=x_range,y=y_pred[0,:,0])
        # # plt.scatter(x=x_range,y=y_true[0,:,0])
        # plt.cla()
        # plt.clf()
        # plt.close()
        # plt.plot(y_pred[0,int(0.75*float(y_pred.shape[0])):,0],'o')
        # plt.plot(y_true[0,int(0.75*float(y_true.shape[0])):,0],'^')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.title('pred vs. y_true')
        # plt.ylabel('crack growth rate, normalized and centered')
        # plt.xlabel('cycles * ' + str(resample_interval))
        # #plt.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
        # plt.legend(['pred[0]', 'true[0]'], loc='upper left')
        #
        # plt.savefig(str(files[0])[:-4] + 'lstm_results_detail_flaw_0' + '.png', bbox_inches='tight')
        # plt.cla()
        # plt.clf()
        # plt.close()
        #
        # # plt.scatter(x= x_range,y=y_pred[0, :, 1])
        # # plt.scatter(x=x_range,y=y_true[0, :, 1])
        # plt.plot(y_pred[0,int(0.75*float(y_pred.shape[0])):,1],'o')
        # plt.plot(y_true[0,int(0.75*float(y_true.shape[0])):,1],'^')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.title('pred vs. y_true')
        # plt.ylabel('crack growth rate, normalized and centered')
        # plt.xlabel('cycles * ' + str(resample_interval))
        # #plt.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
        # plt.legend(['pred[1]', 'true[1]'], loc='upper left')
        #
        # plt.savefig(str(files[0])[:-4] + 'lstm_results_detail_flaw_1' + '.png', bbox_inches='tight')
        # plt.clf()
        # plt.cla()
        # plt.close()
        # # plt.scatter(x= x_range,y=y_pred[0, :, 1])
        # # plt.scatter(x=x_range,y=y_true[0, :, 1])
        # plt.plot(y_pred[0, int(0.75*float(y_pred.shape[0])):, 2], 'o')
        # plt.plot(y_true[0, int(0.75*float(y_true.shape[0])):, 2], '^')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.title('pred vs. y_true')
        # plt.ylabel('crack growth rate, normalized and centered')
        # plt.xlabel('cycles * ' + str(resample_interval))
        # # plt.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
        # plt.legend(['pred[2]', 'true[2]'], loc='upper left')
        #
        # plt.savefig(str(files[0])[:-4] + 'lstm_results_detail_flaw_2' + '.png', bbox_inches='tight')
        # plt.clf()
        # plt.cla()
        # plt.close()
        # # plt.scatter(x= x_range,y=y_pred[0, :, 1])
        # # plt.scatter(x=x_range,y=y_true[0, :, 1])
        # plt.plot(y_pred[0, int(0.75*float(y_pred.shape[0])):, 3], 'o')
        # plt.plot(y_true[0, int(0.75*float(y_true.shape[0])):, 3], '^')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.title('pred vs. y_true')
        # plt.ylabel('crack growth rate, normalized and centered')
        # plt.xlabel('cycles * ' + str(resample_interval))
        # # plt.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
        # plt.legend(['pred[3]', 'true[3]'], loc='upper left')
        #
        # plt.savefig(str(files[0])[:-4] + 'lstm_results_detail_flaw_3' + '.png', bbox_inches='tight')
        # plt.clf()
        # plt.cla()
        # plt.close()
            #print("Score: {}".format(score)) #test_array.shape[0]//generator_batch_size
        # #predictions = model.predict_generator(test_generator, steps=(1*test_array.shape[0]//generator_batch_size),max_queue_size=test_array.shape[0],use_multiprocessing=True)
        # print("scores: {}".format(score))
        # np.savetxt(Base_Path + 'results/TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
        #            fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
        #            footer=str(), comments='# ')


