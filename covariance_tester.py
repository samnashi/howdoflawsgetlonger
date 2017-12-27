import numpy as np
import json
import pandas as pd
import os
from corpus_characterizer import generator_chunker
np.set_printoptions(precision=3,suppress = True, linewidth = 150)

def aggregate_data_and_label(data_array,label_array,desired_colnum = 14):
    assert(data_array.shape[0] == label_array.shape[0])
    array_relevant_vars = np.empty(shape=(data_array.shape[0], desired_colnum))
    array_relevant_vars[:, 0:data_array.shape[1]] = data_array[:,:]
    array_relevant_vars[:, desired_colnum-4:desired_colnum] = label_array[:,1:]
    return array_relevant_vars

import matplotlib.pyplot as plt
#
#
# def estimate_nonlinearity_onset(return_complete = True, array_path = "", num_flaws=4, min_batch_size=128):
#     '''This is designed to be called during the run of the parser, so there's no pre-existing dict
#     to fall back on. '''
#     rates_at_intervals={}
#     array = np.load(array_path) #small enough to not use a generator.
#     largest_multiple_of_batch_size = min_batch_size * (array.shape[0]//min_batch_size) #the row dimension
#     sequence_characteristics = {}
#     for i in range(0, largest_multiple_of_batch_size, min_batch_size):
#         if i % 2*min_batch_size == 0:
#             sequence_characteristics['bsize=' + str(2*min_batch_size)]=1
#             np.cov(array[i,1],array[i,2]) #TODO complete this function call
#             pass
#         if i % 3*min_batch_size == 0:
#             sequence_characteristics['bsize=' + str(3 * min_batch_size)] = 1
#             pass
#         if i % 4*min_batch_size == 0:
#             sequence_characteristics['bsize=' + str(4 * min_batch_size)] = 1
#             pass
#         if i % 5*min_batch_size == 0:
#             sequence_characteristics['bsize=' + str(5 * min_batch_size)] = 1
#             pass

#@@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
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
# analysis_path = "/home/ihsan/Documents/thesis_models/analysis/"
#%%%%%%%%%%%%% TO RUN ON LOCAL (EFI) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Base_Path = "/home/efi/Documents/thesis_models/"
# image_path = "/home/efi/Documents/thesis_models/images"
# train_path = "/home/efi/Documents/thesis_models/train/"
# test_path = "/home/efi/Documents/thesis_models/test/"
# analysis_path = "home/efi/Documents/thesis_models/analysis"
#seq_length_dict_filename = train_path + "/data/seq_length_dict.json"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
desired_colnumber = 15
#TODO: try with 15.
header_corrcoef = "first 9: all but percent_damage and step_index. last four: deltaA 1-2-3-4"

length_total = 0
for index_to_load in range(0,len(combined_filenames)):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    identifier = files[0][:-4]
    length_total += train_array.shape[0]
    #BLOCK FOR AGGREGATED SEQUENCES
print("total length = {}".format(length_total))
temp_array = np.empty(shape=(length_total, desired_colnumber))


list_cov_for_df = []
list_corr_for_df = []
filenames_for_df_index=[]
cov_dict = {}
corr_dict = {}

for index_to_load in range(0,len(combined_filenames)):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    if train_array.shape[1] > 11:
        train_array = train_array[:,1:]
    if label_train_array.shape[1] > 5:
        label_train_array = label_train_array[:,1:]
    identifier = files[0][:-4]

    filenames_for_df_index.append(files[0])

    #BLOCK FOR INDIVIDUAL SEQUENCES
    relevant_vars_array = aggregate_data_and_label(data_array = train_array,
                                                   label_array=label_train_array,
                                                   desired_colnum=desired_colnumber)
    filename_cov = analysis_path + "cov_complete_train_fv1c_" + identifier + ".csv"
    filename_corr = analysis_path + "corrcoef_complete_train_fv1c_" + identifier + ".csv"

    X_cov = np.cov(relevant_vars_array, rowvar=False)
    X_corr = np.corrcoef(relevant_vars_array, rowvar=False)
    cov_dict[files[0]] = X_cov.flatten()
    corr_dict[files[0]] = X_corr.flatten()

    # if index_to_load == 0:
    #     df_cov = pd.DataFrame(X_cov.flatten())
    #     df_corr = pd.DataFrame(X_corr.flatten())
    # if index_to_load != 0:
    #     df_cov_temp = pd.DataFrame(X_cov.flatten())
    #     #print(df_cov_temp.info())
    #     df_corr_temp = pd.DataFrame(X_corr.flatten())
    #     #print(df_corr_temp.info())
    #     print("appending")
    #     df_cov = df_cov.append(df_cov_temp)
    #     df_corr = df_corr.append(df_corr_temp)

    # list_cov_for_df.append(X_cov)
    # list_corr_for_df.append(X_corr)

    #ENABLE THE CODE BELOW TO SAVE THE ARRAYS
    # np.savetxt(fname = filename_cov, X = np.cov(relevant_vars_array,rowvar=False),
    #            delimiter=",",header = header_corrcoef)
    # np.savetxt(fname = filename_corr, X = np.corrcoef(relevant_vars_array,rowvar=False),
    #            delimiter=",",header = header_corrcoef)
    print(("corrcoef and cov for {} saved.").format(identifier))


df_cov = pd.DataFrame.from_dict(cov_dict,orient='index') #keys should be the rows.
df_corr = pd.DataFrame.from_dict(corr_dict,orient='index')
print(df_cov.describe())
print(df_corr.describe())
(df_corr.describe()).to_csv('./df_corr_orientindex_describe.csv')
(df_cov.describe()).to_csv('./df_cov_orientindex_describe.csv')
# df_corr.set_index(filenames_for_df_index) doesn't work
# df_cov.set_index(filenames_for_df_index)
# print("df_corr: {}".format(df_corr.shape)) #describe needs column names
# print("df_cov: {}".format(df_cov.shape))
#df_corr.to_csv('./analysis/df_corr.csv')
#df_cov.to_csv('./analysis/df_cov.csv')
#combined_df = pd.DataFrame(data=[list_cov_for_df,list_corr_for_df]) this makes a 2-row DF...

# cov_df = pd.DataFrame(data=list_cov_for_df)
# corr_df = pd.DataFrame(data=list_corr_for_df)
#print("combined df describe: {}".format(combined_df.describe()))
#
#
# #seq_group_params = {}
# seq_group_params_filename = "./analysis/seq_group_params.json"
# seq_group_params = json.load(open(seq_group_params_filename))
#
# #seq_individual_params = {}
# seq_individual_params_filename = "./analysis/seq_individual_params.json"
# seq_individual_params = json.load(open(seq_individual_params_filename))
#
#
# #seq_entire_params = {}
# seq_entire_params_filename = "./analysis/seq_entire_params.json"
# seq_entire_params = json.load(open(seq_entire_params_filename))
#
# data_array_test_filename = "./train/data/sequence_1b_51_2_fv1b.npy"
# label_array_test_filename = "./train/label/sequence_1b_51_2_label_fv1b.npy"
# data_array_test = np.load(data_array_test_filename)
# label_array_test = np.load(label_array_test_filename)
#
# #onset of linearity is the second derivative. do this numerically.
# #stepindex is kept in the label file.
# array_to_feed = np.empty(shape=(data_array_test.shape[0],2))
# array_to_feed[:,0] = label_array_test[:,0] #stepIndex
# array_to_feed[:,1] = np.log(label_array_test[:,1]) #delta_a_current_1
# first_derivative_deltaA_flaw0 = np.asarray(np.gradient(array_to_feed,axis=0)) #drop the constant first column.
# first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[:,1]
#
# # first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[first_derivative_deltaA_flaw0 == np.inf] = 999
# # first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[first_derivative_deltaA_flaw0 == -np.inf] = 0
# print(first_derivative_deltaA_flaw0)
#
# array_to_feed[:,1] = first_derivative_deltaA_flaw0
# second_derivative_deltaA_flaw0 = np.asarray(np.gradient(array_to_feed,axis=0))
# second_derivative_deltaA_flaw0 = second_derivative_deltaA_flaw0[:,1]
# print(second_derivative_deltaA_flaw0)
# print(np.where(np.diff(np.signbit(second_derivative_deltaA_flaw0)))[0])
# array_to_df = np.empty(shape=(data_array_test.shape[0],2))
# array_to_df[:,0] = label_array_test[:,0]
# array_to_df[:,1] = second_derivative_deltaA_flaw0
#
# second_derivative_df = pd.DataFrame(array_to_df)
# #plot = second_derivative_df.iloc[:,1].plot(logx=True)
# #plt.savefig('./figure_try.png')
#
# cov_flaw0_flaw1 = np.cov(label_array_test[:,1],label_array_test[:,2], rowvar = False) #each COLUMN is a variable
# corrcoef_flaw0_flaw1 = np.cov(label_array_test[:,1],label_array_test[:,2], rowvar = False)
# print(cov_flaw0_flaw1)
# print(corrcoef_flaw0_flaw1)
#
#
# #--------- starts here.
# relevant_vars_array = np.empty(shape=(data_array_test.shape[0],13))
# print("slice of data array shape: {}".format(data_array_test[:,2:].shape))
# relevant_vars_array[:,0:9] = data_array_test[:,2:]
# print("slice of label array shape: {}".format(label_array_test[:,1:].shape))
# relevant_vars_array[:,9:14] = label_array_test[:,1:]
#
# corr_one_sequence = np.cov(data_array_test, rowvar = False) #each COLUMN is a variable
# print(corr_one_sequence.shape)
# print(corr_one_sequence)
#
# corr_one_sequence_combined = np.cov(relevant_vars_array,rowvar = False)
# print(corr_one_sequence_combined.shape)
# print(corr_one_sequence_combined)
#
# corrcoef_one_sequence_combined = np.corrcoef(relevant_vars_array,rowvar = False)
# print(corrcoef_one_sequence_combined.shape)
# print(corrcoef_one_sequence_combined)
#
# #incomplete test script -> PASTE THIS BACK INTO THE covariance method in corpus_characterizer
# '''utility to do chunkwise prediction. chunk mode can be "percent_damage" or "step", array list's length should be 1 or 2.
#  calculate each chunk means calculate every chunk's statistics; calculate cumulative means it's like a partial fit
#  with the temporary states saved in between. This method is used in many other modules, even to compare results'''
# array_list = []
# calculate_each_chunk = True
# calculate_cumulative_each_chunk = True,
# chunks = 5,
# chunk_mode = 'step'
#
# isExhausted = False
# if (type(array_list) != list) or type(array_list[0]) != np.ndarray or type(array_list[1] != np.ndarray):
#     print("Input is of an incorrect type. You need a list of numpy ndarrays.")
#     raise TypeError
# if len(array_list) > 2:
#     print("List of input arrays is longer than 2, only the first 2 arrays will be used.")
# if calculate_cumulative_each_chunk == True and calculate_each_chunk == True:
#     pass
#     #make a two-array generator
# array_list[0]=None
# array_list[1]= None
