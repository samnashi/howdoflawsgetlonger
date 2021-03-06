from __future__ import print_function
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, GRU, Flatten, Input, Reshape, TimeDistributed, Bidirectional, BatchNormalization
from keras.initializers import lecun_normal,glorot_normal
from keras.optimizers import rmsprop
from keras.regularizers import l1,l1_l2,l2
from keras import metrics
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
import sklearn.preprocessing
from keras.callbacks import CSVLogger


'''This is the pure-RNN script with harness. '''

# def batch_size_verifier(generator_batch_size = 64,array_size = 10000,steps_per_epoch=5):
#     limit = generator_batch_size * (array_size//generator_batch_size)
#     minimum_batch_size =



#you limit the # of calls keras calls the generator OUTSIDE the generator.
#each time you fit, dataset length // batch size. round down!

def pair_generator_lstm(data, labels, start_at=0, generator_batch_size=64, scaled=True, scaler_type ='standard',
                        scale_what ='data',use_precomputed_coeffs = False, label_dims = 4, no_labels=False): #shape is something like 1, 11520, 11
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
            #print('Standard Scaler initialized \n')
        elif scaler_type == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
            label_scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = sklearn.preprocessing.RobustScaler()
            label_scaler = sklearn.preprocessing.RobustScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler() #TRY NORMALIZER FOR THE LABEL
        #print("scaled: {}, scaler_type: {}".format(scaled, scaler_type))\
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
            data_scaled = scaler.transform(X=data)
            labels_scaled = label_scaler.transform(X=labels)
        if use_precomputed_coeffs == False:
            data_scaled = scaler.fit_transform(X=data)
            labels_scaled = scaler.fit_transform(X=labels) #i don't think you should scale the labels..
        #labels_scaled = labels
        # data_scaled = np.reshape(data_scaled,(1,data_scaled.shape[0],data_scaled.shape[1]))
        # labels_scaled = np.reshape(labels_scaled, (1, labels_scaled.shape[0],labels_scaled.shape[1]))

        if data_scaled.shape[1] != 11: #TODO properly investigate why some data arrays have 12 columns..
            data_scaled = data_scaled[:, 1:]

        data_scaled = np.expand_dims(data_scaled, axis=0)  # add 1 dimension in the 0th axis
        labels_scaled = np.expand_dims(labels_scaled, axis=0)
        index = start_at

    while 1:
        #if index < ((data_scaled.shape[1]-start_at)//generator_batch_size)* generator_batch_size:  # for index in range(start_at,generator_batch_size*(data.shape[1]//generator_batch_size)):
        #while index < ((data_scaled.shape[1]-start_at)//generator_batch_size)* generator_batch_size: # for index in range(start_at,data_scaled.shape[1]):
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x = (data_scaled[:, index:index + generator_batch_size, :])  # first dim = 0 doesn't work.

        if labels_scaled.shape[2] > label_dims:
            #cut the step_index. it's like the sprue to ensure rigidity of data.
            y = (labels_scaled[:, index:index + generator_batch_size, 1:])
        if labels_scaled.shape[2] == label_dims:
            y = (labels_scaled[:, index:index + generator_batch_size, :])

        # OLD y = (labels_scaled[:, index:index + generator_batch_size, :])  # yield shape = (4,)


        #generator_batch_size * (data_scaled.shape[1] - start_at) // generator_batch_size
        if index + 2 * generator_batch_size < data_scaled.shape[1]:
            index = index + generator_batch_size
        else:
            #index = np.random.randint(low=0,high=(generator_batch_size*((data_scaled.shape[1]-start_at)//generator_batch_size-2)))
            index = np.random.randint(low=max(0,(generator_batch_size*((data_scaled.shape[1]-start_at)//generator_batch_size-10))),high=(generator_batch_size*((data_scaled.shape[1]-start_at)//generator_batch_size-2)))
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
        if no_labels == False:
            yield (x, y)
        if no_labels == True:
            yield(x)


def reference_bilstm_nodense_micro(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False,
                                   imp=2, dense_act='tanh'):
    '''reference SMALL BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(32, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(32, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_bilstm_nodense_tiny(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False,
                                  imp=2, dense_act='tanh'):
    '''reference TINY BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(64, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_bilstm_nodense_medium(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(),
                                    sf=False, imp=2, dense_act='tanh'):
    '''reference MEDIUM BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(100, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_bilstm_nodense_big(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(),
                                    sf=False, imp=2, dense_act='tanh'):
    '''reference SMALL BiLSTM with batchnorm and TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_bilstm_big_dense(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False, imp=2,
                         dense_act='tanh'):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(input_tensor)
    i = BatchNormalization()(h)
    j = Bidirectional(LSTM(200, kernel_initializer=k_init, return_sequences=True,
                           recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
                           implementation=imp, stateful=sf))(i)
    j = BatchNormalization()(j)
    k = TimeDistributed(Dense(64, kernel_initializer=k_init, activation=dense_act,
                              kernel_regularizer=k_reg))(j)
    # l = BatchNormalization()(k)
    # out = Dense(4)(l)
    out = k
    return out

def reference_lstm_nodense_micro(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False,
                                 imp=2):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(32, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(32, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_nodense_tiny(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False,
                                imp=2):
    '''reference BiLSTM with batchnorm and elu TD-dense.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(64, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(64, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

def reference_lstm_nodense_medium(input_tensor, k_init=lecun_normal(seed=1337), k_reg=l1(), rec_reg=l1(), sf=False,
                                  imp=2):
    '''reference BiLSTM with batchnorm and NO dense layers. Output is already batch-normed.
    Expects ORIGINAL input batch size so pad/adjust window size accordingly!'''
    h = LSTM(100, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(input_tensor)
    i = BatchNormalization()(h)
    j = LSTM(100, kernel_initializer=k_init, return_sequences=True,
             recurrent_regularizer=rec_reg, kernel_regularizer=k_reg,
             implementation=imp, stateful=sf)(i)
    j = BatchNormalization()(j)
    out = j
    return out

if __name__ == "__main__":

    #!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
    num_epochs = 3 #individual. like how many times is the net trained on that sequence consecutively
    num_sequence_draws = 100 #how many times the training corpus is sampled.
    generator_batch_size = 128
    finetune = False
    test_only = False #no training. if finetune is also on, this'll raise an error.
    use_precomp_sscaler = True
    sequence_circumnavigation_amt = 0.7
    save_preds = True
    save_figs = False
    identifier_pre_training = '1200_small_l1_' #weights to initialize with, if fine tuning is on.
    identifier_post_training = "_big_bidir_" #weight name to save as
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
    #load data multiple times.
    data_filenames = list(set(os.listdir(train_path + "data")))
    #print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    #print("after sorting, data_filenames: {}".format(data_filenames))

    label_filenames = list(set(os.listdir(train_path + "label")))
    label_filenames.sort() #sorting makes sure the label and the data are lined up.
    #print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames,label_filenames)
    #print("before shuffling: {}".format(combined_filenames))
    shuffle(combined_filenames)
    print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.

    #define the model first
    #TODO function def to create LSTM model
    #TODO boolean flags for retraining
    #TODO: check if model runs just fine with the batch norm layers

    kr = l1_l2()
    a = Input(shape=(None,11))
    lstm = reference_bilstm_nodense_big(input_tensor=a, k_reg=kr, k_init='orthogonal', rec_reg=kr)
    d = TimeDistributed(Dense(64,activation='tanh',kernel_initializer=lecun_normal(seed=1337),kernel_regularizer=kr,name='dense_post_concat'),)(lstm)
    e = BatchNormalization(name='bn_final')(d)
    out = TimeDistributed(Dense(4,kernel_initializer=lecun_normal(seed=1337)))(e)

    keras_optimizer = rmsprop(lr=0.0015, rho=0.9, epsilon=1e-08, decay=0.0)
    model = Model(inputs=a,outputs=out)
    metrics_list = ['mae', 'mape', 'mse', 'msle']
    model.compile(loss='mse', optimizer=keras_optimizer,metrics=metrics_list)
    print("Model summary: {}".format(model.summary()))
    print("Inputs: {}".format(model.input_shape))
    print ("Outputs: {}".format(model.output_shape))
    print ("Metrics: {}".format(model.metrics_names))

    plot_model(model, to_file = analysis_path + 'model_' + identifier_post_training + '.png', show_shapes=True)
    #print ("Actual input: {}".format(data.shape))
    #print ("Actual output: {}".format(target.shape))
    weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
    print('Data loaded.')
    if finetune==False:
        weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
        print("Are weights (with the given name to be saved as) already present? {}".format(weights_present_indicator))
    else:
        weights_present_indicator = os.path.isfile('Weights_' + identifier_pre_training + '.h5')
        print("Are weights (with the given name) to initialize with present? {}".format(weights_present_indicator))

    csv_logger = CSVLogger(filename = './analysis/logtrain' + identifier_post_training + ".csv", append=True)
    weights_file_name = None

    if (finetune == False and weights_present_indicator == False and test_only == False) or (
            finetune == True and weights_present_indicator == True):
        print("TRAINING PHASE")
        print("weights_present_indicator: {}, finetune: {}".format(weights_present_indicator,finetune))
        for i in range(0,num_sequence_draws):
            index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
            files = combined_filenames[index_to_load]
            data_load_path = train_path + '/data/' + files[0]
            label_load_path = train_path + '/label/' + files[1]
            #print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
            train_array = np.load(data_load_path)
            label_array = np.load(label_load_path)[:,1:] #,1: is an artifact, StepIndex is dropped, was only there to ensure rigidity
            if train_array.shape[1] != 11:
                train_array=train_array[:,1:]
            #-----------COMMENT THESE OUT IF YOU WANT RESCALER ON----------------------------------
            #train_array = np.reshape(train_array,(1,train_array.shape[0],train_array.shape[1]))
            #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!
            #--------------------------------------------------------------------------------------
            print("filename: {}, data/label shape: {}, {}, "
                  "draw #: {} out of {}".format(str(files[0]),train_array.shape,label_array.shape, i, num_sequence_draws))
            nonlinear_part_starting_position = generator_batch_size * ((train_array.shape[0] // generator_batch_size) - 5)
            shuffled_starting_position = np.random.randint(0,nonlinear_part_starting_position)

            if finetune == True: #load the weights
                finetune_init_weights_filename = 'Weights_' + str(500) + identifier_pre_training + '.h5'
                model.load_weights(finetune_init_weights_filename, by_name=True)

            train_generator = pair_generator_lstm(train_array, label_array, start_at=shuffled_starting_position,
                                generator_batch_size=generator_batch_size, use_precomputed_coeffs=False,label_dims=4)

            if i == 0 :
                training_hist = model.fit_generator(train_generator,epochs=num_epochs,
                                                    steps_per_epoch= 1 * (train_array.shape[0] // generator_batch_size),
                                                    callbacks = [csv_logger], verbose=2)
            else:
                training_hist_increment = model.fit_generator(train_generator, epochs=num_epochs,
                                                              steps_per_epoch= 1 * (train_array.shape[0] // generator_batch_size),
                                                              callbacks = [csv_logger],verbose=2)

        if weights_present_indicator == True and finetune == True:
            print("fine-tuning/partial training session completed.")
            weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
            model.save_weights(weights_file_name)
            model.save('./analysis/model_lstm_' + identifier_post_training + '.h5')
            print("after {} iterations, model weights is saved as {}".format(num_sequence_draws * num_epochs,
                                                                             weights_file_name))
        if weights_present_indicator == False and finetune == False:  # fresh training
            print("FRESH training session completed.")
            weights_file_name = 'Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'
            model.save_weights(weights_file_name)
            model.save('./analysis/model_lstm_' + identifier_post_training + '.h5')
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
        #test_weights_present_indicator = True
    print("weights_file_name after the if/else block to determine the test flag is: {}".format(weights_file_name))
    print("test_weights_indicator is: {}".format(test_weights_present_indicator))

    if test_weights_present_indicator == True:
        #the testing part
        print("TESTING PHASE, with weights: {}".format('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5'))
        #print("TESTING PHASE, with weights {}".format('Weights_300_3_firstrun_fv1b_server'))
        model.load_weights('Weights_' + str(num_sequence_draws) + identifier_post_training + '.h5')
        #model.load_weights('Weights_300_3_firstrun_fv1b_server.h5')

        # load data multiple times.

        #data_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/data/')
        data_filenames = list(set(os.listdir(test_path + "data")))
        # print("before sorting, data_filenames: {}".format(data_filenames))
        data_filenames.sort()
        # print("after sorting, data_filenames: {}".format(data_filenames))

        #label_filenames = os.listdir('/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/label')
        label_filenames = list(set(os.listdir(test_path + "label")))
        label_filenames.sort()
        # print("label_filenames: {}".format(data_filenames))
        assert len(data_filenames) == len(label_filenames)
        combined_filenames = zip(data_filenames, label_filenames)
        # print("before shuffling: {}".format(combined_filenames))
        shuffle(combined_filenames)
        print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.

        i=0
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
            if test_array.shape[1] != 11:
                test_array=test_array[:,1:]
            #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
            #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
            #label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label doesn't need to be 3D
            print("filename: {}, data/label shape: {}, {}".format(str(files[0]),test_array.shape, label_array.shape))
            predictions_length = generator_batch_size * (label_array.shape[0]//generator_batch_size)
            #largest integer multiple of the generator batch size that fits into the length of the sequence.
            #TODO update to new API regarding initialization of precomputed coeffs.
            test_generator = pair_generator_lstm(test_array, label_array, start_at = 0,
                                                 generator_batch_size=generator_batch_size, use_precomputed_coeffs = False,label_dims=4)
            row_dict = {}
            score = model.evaluate_generator(test_generator, steps=(1 * test_array.shape[0] // generator_batch_size),
                                             max_queue_size=test_array.shape[0], use_multiprocessing=False)
            print("scores: {}".format(score))

            #metrics=['accuracy','mae','mape','mse'])
            metrics_check = (metrics_list == model.metrics_names)
            if metrics_check == False:
                metrics_list = model.metrics_names
            #print("is metrics_list equal to the score dict's key?", metrics_list == model.metrics_names)

            row_dict['filename'] = str(files[0])[:-4]
            for item in metrics_list:
                row_dict[str(item)] = score[metrics_list.index(item)] #'loss'
            # row_dict['mae'] = score[1] #'acc'
            # row_dict['mape'] =score[2] #'mean_absolute_error'
            # row_dict['mse'] = score[3] #'mean_absolute_percentage_error'
            score_rows_list.append(row_dict)

            test_generator = pair_generator_lstm(test_array, label_array, start_at=0,
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
                np.save(Base_Path + 'preds_' + identifier_post_training + "_" + str(files[0]), arr=y_prediction)

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
            #scaler_output = set_standalone_scaler_params(scaler_output)
            # print("")
            #label_truth = scaler_output.transform(X=label_truth_temp)

            resample_interval = 16
            label_truth = label_truth[::resample_interval, :]
            y_prediction = y_prediction[::resample_interval, :]

        score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())
        score_df.to_csv('scores_lstm_rerun_' + identifier_post_training + '.csv')

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

