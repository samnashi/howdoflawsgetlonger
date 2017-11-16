from __future__ import print_function
import numpy as np
from random import shuffle
import pandas as pd
import scipy.io as sio
import os
import json
import scattergro_utils as sg_utils
from sklearn.preprocessing import StandardScaler
from scattergro_parser_each import parse_scattergro
import matplotlib.pyplot as plt
import pprint
from sklearn.metrics import mean_squared_error, mean_absolute_error


'''Characterizes sequences. Calculates the relevant statistics.'''

#THIS IS USED IN CHUNKER_TESTER
def generator_chunker(array_raw, chunker_batch_size, start_at = 0,scaler_active=True,scaler_type='standard_per_batch'):
    '''This is designed as a tool to ease comparisons between really large arrays.
    Use case: define one generator to load and yield chunks of a prediction array,
    then define another generator to load and yield chunks of the labels/targets.
     Feed the results of both generator's .next() into a scikit-learn regression loss function.'''
    #find largest multiple of the batch size that'd go into the array. largest common multiple
    assert type(array_raw) == np.ndarray
    lcm = chunker_batch_size * (array_raw.shape[0] // chunker_batch_size)
    array_trimmed = array_raw[0:lcm,:]
    #cut down the array
    index = start_at #initialize
    if scaler_active == True:
        scaler = StandardScaler()
    while 1:
        x = array_trimmed[index:index+chunker_batch_size,:]
        index = index+chunker_batch_size
        assert x.shape[0]==chunker_batch_size
        if scaler_active == True and scaler_type=='standard_per_batch':
            x = scaler.fit_transform(x)
        yield x

#TODO pseudocode in green notebook. complete this before the 29th
def estimate_nonlinearity_onset(return_complete = True, array_path = "", num_flaws=4, min_batch_size=128):
    '''This is designed to be called during the run of the parser, so there's no pre-existing dict
    to fall back on. '''
    rates_at_intervals={}
    array = np.load(array_path) #small enough to not use a generator.
    largest_multiple_of_batch_size = min_batch_size * (array.shape[0]//min_batch_size) #the row dimension
    sequence_characteristics = {}
    for i in range(0, largest_multiple_of_batch_size, min_batch_size):
        if i % 2*min_batch_size == 0:
            sequence_characteristics['bsize=' + str(2*min_batch_size)]=1
            np.cov(array[i,1],array[i,2])
            pass
        if i % 3*min_batch_size == 0:
            sequence_characteristics['bsize=' + str(3 * min_batch_size)] = 1
            pass
        if i % 4*min_batch_size == 0:
            sequence_characteristics['bsize=' + str(4 * min_batch_size)] = 1
            pass
        if i % 5*min_batch_size == 0:
            sequence_characteristics['bsize=' + str(5 * min_batch_size)] = 1
            pass


    # TODO calculate covariances of everything.
    #possible_combinations = combinations_with_replacement(  # column numbers ,r=2)
    # crack position vs crack growth rate

    # load vs. crack growth rate

    # TODO find the kink in crack growth rate.
    # probably the correlation between the load and the crack growth rate, on each crack..
    # use pearson_r
    return rates_at_intervals

def compare_standardscaler_coeffs_from_json(fname_entire ="" ,fname_group="",fname_individual="",calculate_all = True):
    '''returns a dict of lists. This method is to analyze teh variation of standardscaler coefficients, and hopefully
    be able to come up with a satisfactory set of SS parameters that works for both training and testing'''

    standardscaler_coeffs_dict = {}
    seq_entire_params_filename = fname_entire
    if fname_entire == "":
        seq_entire_params_filename = "./analysis/seq_entire_params.json"
    seq_entire_params = json.load(open(seq_entire_params_filename))

    #usage print(seq_entire_params[seq_entire_params.iterkeys().next()].keys())

    seq_group_params_filename = fname_group
    if fname_group == "":
        seq_group_params_filename = "./analysis/seq_group_params.json"
    seq_group_params = json.load(open(seq_group_params_filename))

    seq_individual_params_filename = fname_individual
    if fname_individual == "":
        seq_individual_params_filename = "./analysis/seq_individual_params.json"
    seq_individual_params = json.load(open(seq_individual_params_filename))

    if calculate_all == True:
        '''right now, all this does is apply one aggregation method function on each dict, but this can be tweaked later '''
        # entire
        scale_entire_data_list = [] #this is the final values. cast it to a numpy array before returning.
        scale_entire_label_list = []
        #first item's shape. should be 11 cols.
        scale_entire_data_np = np.empty(shape = (np.asarray(seq_entire_params['data']['scale']).shape))
        scale_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['scale']).shape))

        mean_entire_data_list = []
        mean_entire_label_list = []
        mean_entire_data_np = np.empty(shape = (np.asarray(seq_entire_params['data']['mean']).shape))
        mean_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['mean']).shape))

        var_entire_data_list = []
        var_entire_label_list = []
        var_entire_data_np = np.empty(shape = (np.asarray(seq_entire_params['data']['var']).shape))
        mean_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['mean']).shape))


        for key_entire, values in seq_entire_params: #data or label 
            for key_tier2 in values:
                if key_entire == 'data':
                    if key_tier2 == 'scale':
                        scale_entire_data_list.append(seq_entire_params[key_tier2])
                    if key_tier2 == 'mean':
                        mean_entire_data_list.append(seq_entire_params[key_tier2])
                    if key_tier2 == 'var':
                        var_entire_data_list.append(seq_entire_params[key_tier2])

        var_entire_data_np = np.array(var_entire_data_list)
        assert var_entire_data_np.shape == np.asarray(seq_entire_params['data']['var']).shape

        #mean_entire_data
        scale_group_data_list = []
        mean_group_data_list = []
        var_group__data_list = []
        scale_group_label_list = []
        mean_group_label_list = []
        var_group__label_list = []
        for keys_group in seq_group_params.keys(): #these are csv names.
            scale_group_data_np = np.empty(shape = (np.asarray(seq_group_params['train'][''])))

            #still need to go through the second tier
            #TODO: group dict is also useful for finding out which group does the sequence belong to, for postmortem
            pass

        scale_individual_list = []
        mean_individual_list = []
        var_individual_list = []
        for keys_group in seq_group_params.keys():
            #still need to go through
            pass
    #TODO numpy savetxt covariance matrices

def training_data_characterizer(data_path = "./train/data/",label_path = "./train/label/",analysis_path = "./analysis/", analysis_mode=True):
    individual_sequence_scaler = StandardScaler() #to fit and save on each.
    individual_label_scaler = StandardScaler()
    all_train_data_sequence_scaler = StandardScaler() #to be partial fitted
    all_labels_scaler = StandardScaler()
    data_folder_contents_raw = os.listdir(data_path)
    data_folder_contents_filtered = []
    for data_sequence in data_folder_contents_raw:
        if data_sequence.endswith(".npy"):     #make sure it's only .npy files that are being processed
            data_folder_contents_filtered.append(data_sequence)
    label_folder_contents_raw = os.listdir(label_path)
    label_folder_contents_filtered = []
    for label_sequence in label_folder_contents_raw:
        if label_sequence.endswith(".npy"):
            label_folder_contents_filtered.append(label_sequence)
        pass
    #inplace conversion to absolute paths
    for item in data_folder_contents_filtered:
        item = data_path + str(item)
    for label in label_folder_contents_filtered:
        label = label_path + str(label)
    # ------------ANALYSIS PART-----------------------------------------------------------------------------
    if analysis_mode == True:  # calculates statistics
        # calculates the characteristic parameters of blocks of sequences (same IC and same load cond)
        individual_sequence_scaler_params = {}
        individual_label_scaler_params = {}

if __name__ == "__main__":
    '''    processed_path = "/home/ihsan/Documents/thesis_models/unsplit"
    path = "/home/ihsan/Documents"
    processed_path = "/home/ihsan/Documents/thesis_models/unsplit"
    seq_length_dict = {}
    seq_length_dict_filename = processed_path + "/sequence_lengths.json"
    seq_group_params = {}
    seq_group_params_filename = "./analysis/seq_group_params.json"
    seq_individual_params = {}
    seq_individual_params_filename = "./analysis/seq_individual_params.json"
    seq_entire_params = {}
    seq_entire_params_filename = "./analysis/seq_entire_params.json"'''

    #this is the tester for the read json part.
    seq_entire_params_filename = "./analysis/seq_entire_params.json"
    seq_group_params_filename = "./analysis/seq_group_params.json"
    seq_individual_params_filename = "./analysis/seq_individual_params.json"


    seq_entire_params = json.load(open( seq_entire_params_filename))
    seq_group_params = json.load(open( seq_group_params_filename))
    seq_individual_params = json.load(open( seq_individual_params_filename))

    pp = pprint.PrettyPrinter(indent=1)

    # pp.pprint("entire: {}".format(seq_entire_params))
    # pp.pprint("group: {}".format(seq_group_params))
    # pp.pprint("individual: {}".format(seq_individual_params))

    print("ENTIRE:")
    pp.pprint(seq_entire_params)
    print("GROUP:")
    pp.pprint(seq_group_params)
    print("INDIVIDUAL:")
    pp.pprint(seq_individual_params)
    # print(json.dumps(seq_individual_params,indent=4))

    #TODO bind list positions to the actual column names
    # the order of creation is actually group -> individual -> entire.
    # that's why the colnames are in the "group" dict.
    train_colnames = []
    train_colnames = seq_group_params['train_colnames']
    label_colnames = []
    label_colnames = seq_group_params['label_colnames']

    ''' u'train_colnames': [u'StepIndex',
                     u'percent_damage',
                     u'delta_K_current_1',
                     u'ctip_posn_curr_1',
                     u'delta_K_current_2',
                     u'ctip_posn_curr_2',
                     u'delta_K_current_3',
                     u'ctip_posn_curr_3',
                     u'delta_K_current_4',
                     u'ctip_posn_curr_4',
                     u'Load_1',
                     u'Load_2']}'''

    #TODO: need an if to determine whether this is a label or a data file.

    #EXAMPLE USAGE OF DICT STRUCTURE
    print(seq_individual_params.keys()[0])
    print(seq_individual_params[seq_individual_params.keys()[0]]['scale'])
    print(seq_individual_params[seq_individual_params.keys()[0]]['scale'][train_colnames.index('delta_K_current_1')])



    #TODO load the jsons, plot the damn things.
    #histogram of means.


#call parser
#parse_scattergro(analysis_mode = True, save_arrays = False, feature_identifier = 'fvx')

#stack all the relevant columns of all the sequences
#do a standardscaler partial_fit to calculate the statistics..


'''
#!!!!!!!!!!!!!!!!!!!!!TRAINING SCHEME PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shortest_length = sg_utils.get_shortest_length()  #a suggestion. will also print the remainders.
num_epochs = 3 #individual. like how many times is the net trained on that sequence consecutively
num_sequence_draws = 300 #how many times the training corpus is sampled.
generator_batch_size = 128
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

identifier = "_fv1a_"
Base_Path = "./"
#train_path = "/home/ihsan/Documents/thesis_models/train/"
#test_path = "/home/ihsan/Documents/thesis_models/test/"
train_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/train/'
test_path = '/media/ihsan/BigRigData/Thesis/Dataset_FV1_stepindex/test/'
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#11 input columns
#4 output columns.

np.random.seed(1337)
data_filenames = os.listdir(train_path + "data")
data_filenames.sort()

label_filenames = os.listdir(train_path + "label")
label_filenames.sort() #sorting makes sure the label and the data are lined up.
#print("label_filenames: {}".format(data_filenames))
assert len(data_filenames) == len(label_filenames)
combined_filenames = zip(data_filenames,label_filenames)
#print("before shuffling: {}".format(combined_filenames))
#shuffle(combined_filenames)
#print("after shuffling: {}".format(combined_filenames)) #shuffling works ok.


#weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#HARDCODED

list_of_train_rows = []
row_dict = {}
weights_present_indicator = False
if weights_present_indicator == False:
    print("TRAINING PHASE")

    for item in combined_filenames:
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
        print("filename: {}, data/label shape: {}, {}".format(str(files[0]),train_array.shape,label_array.shape))
        row_dict = {}
        row_dict['filename'] = str(files[0])
        row_dict['length'] = train_array.shape[1]
        row_dict['shape'] = str(train_array.shape)
        list_of_train_rows.append(row_dict)

    train_rows_df = pd.DataFrame(list_of_train_rows,columns = ['filename', 'length','shape'])
    train_rows_df.to_csv('train_characteristics' + identifier + '.csv')
#weights_present_indicator = os.path.isfile('Weights_' + str(num_sequence_draws) + identifier + '.h5')
#-------------------------TESTING PHASE-------------------------------------------------------------
list_of_test_rows = []
row_dict = {}
weights_present_indicator = True
if weights_present_indicator == True:
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
    #shuffle(combined_filenames)

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
        row_dict = {}
        row_dict['filename'] = str(files[0])
        row_dict['length'] = test_array.shape[1]
        row_dict['shape'] = str(test_array.shape)
        score_rows_list.append(row_dict)

    test_rows_df = pd.DataFrame(score_rows_list, columns=score_rows_list[0].keys())
    test_rows_df.to_csv('test_characteristics_' + identifier + '.csv')

    #score_df = pd.DataFrame(data=score_rows_list, columns=score_rows_list[0].keys())


'''