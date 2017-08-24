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


data_path = "./train/data/"
label_path = "./train/label/"
analysis_path = "./analysis/"
analysis_mode=True

# individual_data_sequence_scaler = StandardScaler() #to fit and save on each.
# individual_label_scaler = StandardScaler()
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
for index, item in enumerate(data_folder_contents_filtered):
    data_folder_contents_filtered[index] = data_path + str(item)
for index, item in enumerate(label_folder_contents_filtered):
    label_folder_contents_filtered[index] = label_path + str(item)
print(data_folder_contents_filtered)
print(label_folder_contents_filtered)

#predeclare np arrays to store this stuff in
sample_data_array = np.load(data_folder_contents_filtered[1])
aggregated_data_coeffs_individual = np.empty(shape=(len(data_folder_contents_filtered)*3,sample_data_array.shape[1]-1))
#since i'm saving everything in one array for now. #TODO: MAKE THIS INTO THREE ARRAYS )*3 part
#standardscaler isn't precomputed on stepindex.. any large-scale aggregation on that column makes zero sense! 
aggregated_data_coeffs_group = np.empty(shape=(3,sample_data_array.shape[1]-1)) #TODO OY CHECK THIS 3, what do you actually want to fit anyway?

sample_label_array = np.load(label_folder_contents_filtered[1])
aggregated_label_coeffs_individual = np.empty(shape=(len(data_folder_contents_filtered)*3,sample_label_array.shape[1]-1))
#standardscaler isn't precomputed on stepindex.. any large-scale aggregation on that column makes zero sense! hence the dropped columns. 
aggregated_label_coeffs_group = np.empty(shape=(3,sample_label_array.shape[1]-1)) #TODO OY CHECK THIS 3, what do you actually want to fit anyway?

#save a text file saying what on earth does column mean (e.g. where everything is!)

#serial, what the heck.
i=0
for item in data_folder_contents_filtered:
    individual_data_sequence_scaler = StandardScaler()  # to fit and save on each.
    data_np = np.load(item)
    all_train_data_sequence_scaler.partial_fit(data_np[:,1:])
    individual_data_sequence_scaler.fit(data_np[:,1:])
    aggregated_data_coeffs_individual[i,:] = individual_data_sequence_scaler.mean_
    aggregated_data_coeffs_individual[i+1, :] = individual_data_sequence_scaler.scale_
    aggregated_data_coeffs_individual[i+2, :] = individual_data_sequence_scaler.var_
    # so, every 3 rows is one sequence. Every 2nd row is the scale, every 3rd is the mean etc.
    i += 1 #shorthand for i = i + 1
    #np.save(file="./analysis/")

np.save(analysis_path + "agg_data_coeffs_indiv.npy",aggregated_data_coeffs_individual)
#TODO add headers
#alphabetical order. mean-scale-var.
aggregated_data_coeffs_group[0,:] = np.ndarray.tolist(all_train_data_sequence_scaler.mean_)
aggregated_data_coeffs_group[1,:] = np.ndarray.tolist(all_train_data_sequence_scaler.scale_)
aggregated_data_coeffs_group[2,:] = np.ndarray.tolist(all_train_data_sequence_scaler.var_)

np.savetxt(analysis_path + "agg_data_coeffs_group.txt",aggregated_data_coeffs_group)

i=1
for item in label_folder_contents_filtered:
    individual_label_scaler = StandardScaler()
    label_np = np.load(item)
    all_labels_scaler.partial_fit(label_np[:,1:])
    individual_label_scaler.fit(label_np[:,1:])
    aggregated_label_coeffs_individual[i,:] = individual_label_scaler.mean_
    aggregated_label_coeffs_individual[i+1, :] = individual_label_scaler.scale_
    aggregated_label_coeffs_individual[i+2, :] = individual_label_scaler.var_
    i += 1
    # so, every 3 rows is one sequence. Every 2nd row is the scale, every 3rd is the mean etc.
np.save(analysis_path + "agg_label_coeffs_indiv.npy",aggregated_label_coeffs_individual)
#TODO add headers
aggregated_label_coeffs_group[0,:] = np.ndarray.tolist(all_labels_scaler.mean_)
aggregated_label_coeffs_group[1,:] = np.ndarray.tolist(all_labels_scaler.scale_)
aggregated_label_coeffs_group[2,:] = np.ndarray.tolist(all_labels_scaler.var_)
np.savetxt(analysis_path + "agg_label_coeffs_group.txt",aggregated_label_coeffs_group)

# ------------ANALYSIS PART-----------------------------------------------------------------------------
# if analysis_mode == True:  # calculates statistics
#     # calculates the characteristic parameters of blocks of sequences (same IC and same load cond)
#     individual_sequence_scaler_params = {}
#     individual_label_scaler_params = {}
#
#
#
#     individual_sequence_scaler.partial_fit(train_df_as_np_array)
#     individual_label_scaler.fit(label_train_df_as_np_array)
#     # print(individual_sequence_scaler.mean_, individual_sequence_scaler.scale_, individual_sequence_scaler.var_, individual_sequence_scaler.std_)
#
#     individual_sequence_scaler_params['mean'] = np.ndarray.tolist(individual_sequence_scaler.mean_)
#     individual_sequence_scaler_params['scale'] = np.ndarray.tolist(individual_sequence_scaler.scale_)
#     # individual_sequence_scaler_params['std'] = np.ndarray.tolist(individual_sequence_scaler.std_)
#     individual_sequence_scaler_params['var'] = np.ndarray.tolist(individual_sequence_scaler.var_)
#
#     individual_label_scaler_params['mean'] = np.ndarray.tolist(individual_label_scaler.mean_)
#     individual_label_scaler_params['scale'] = np.ndarray.tolist(individual_label_scaler.scale_)
#     # deprecated individual_label_scaler_params['std'] = np.ndarray.tolist(individual_label_scaler.std_)
#     individual_label_scaler_params['var'] = np.ndarray.tolist(individual_label_scaler.var_)
#
#     # nested dict.
#     seq_individual_params["sequence_" + identifier + "_" + str(j) + "_" + str(
#         i) + ".npy"] = individual_sequence_scaler_params
#     seq_individual_params["sequence_" + identifier + "_" + str(j) + "_" + str(
#         i) + "_label_.npy"] = individual_label_scaler_params
#     # ------------END OF ANALYSIS PART----------------------------------------------------------------------
# if save_arrays == True:
#     np.save(np_train_path, train_df_as_np_array)
#     np.save(np_label_train_path, label_train_df_as_np_array)
# j = j + 1
#
# print(seq_length_dict)  # these are of individual sequence lengths.
# # ---------------ANALYSIS OF UNSPLIT---------------------------------------------------------------------
# if analysis_mode == True:
# # processed_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/processed/'
# items_processed = os.listdir(processed_path)
# items_processed.sort()
# print(type(items_processed))
# for file_p in items_processed:
#     if ('.npy') not in str(file_p):
#         del items_processed[items_processed.index(file_p)]  # get rid of non .npy files from this list.
# print(items_processed)
#
# # run standardscaler on all the sequences. Would be unproductive to do it earlier.
# entire_data_scaler = StandardScaler()
# entire_label_scaler = StandardScaler()
#
# entire_data_scaler_params = {}
# entire_label_scaler_params = {}
#
# for file_p in items_processed:  # TODO these are all tuples..
#     if ("label") in str(file_p):
#         partial_label = np.load(processed_path + '/' + str(file_p))
#         entire_label_scaler.partial_fit(partial_label)
#     if ("label") not in str(file_p):
#         partial_data = np.load(processed_path + '/' + str(file_p))
#         entire_data_scaler.partial_fit(partial_data)
#
# entire_data_scaler_params['mean'] = np.ndarray.tolist(entire_data_scaler.mean_)
# entire_data_scaler_params['scale'] = np.ndarray.tolist(entire_data_scaler.scale_)
# # entire_data_scaler_params['std'] = np.ndarray.tolist(entire_data_scaler.std_)
# entire_data_scaler_params['var'] = np.ndarray.tolist(entire_data_scaler.var_)
#
# entire_label_scaler_params['mean'] = np.ndarray.tolist(entire_label_scaler.mean_)
# entire_label_scaler_params['scale'] = np.ndarray.tolist(entire_label_scaler.scale_)
# # entire_label_scaler_params['std'] = np.ndarray.tolist(entire_label_scaler.std_)
# entire_label_scaler_params['var'] = np.ndarray.tolist(entire_label_scaler.var_)
# seq_entire_params['data'] = entire_data_scaler_params
# seq_entire_params['label'] = entire_label_scaler_params
#
# # TODO calculate covariances of everything.
# # possible_combinations = combinations_with_replacement(#column numbers ,r=2)
# # crack position vs crack growth rate
# # load vs. crack growth rate
#
# # TODO find the kink in crack growth rate.
# # probably the correlation between the load and the crack growth rate, on each crack..
# # use pearson_r
#
#
# # ---------------END OF ANALYSIS---------------------------------------------------------------------
# # TODO use DictWriter to get csvs.
# json.dump(seq_length_dict, open(seq_length_dict_filename, 'wb'))
# json.dump(seq_group_params, open(seq_group_params_filename, 'wb'))
# json.dump(seq_individual_params, open(seq_individual_params_filename, 'wb'))
# json.dump(seq_entire_params, open(seq_entire_params_filename, 'wb'))