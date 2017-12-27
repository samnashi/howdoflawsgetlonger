from corpus_characterizer import generator_chunker
import numpy as np
import os
# from sklearn.metrics import mean_squared_error,mean_absolute_error, median_absolute_error, mean_squared_log_error, explained_variance_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, \
    r2_score
import pandas as pd
from scipy.stats import describe, kurtosistest, skewtest, normaltest
from Conv1D_LSTM_Ensemble import pair_generator_1dconv_lstm_bagged
from Conv1D_ActivationSearch_BigLoop import pair_generator_1dconv_lstm #NOT BAGGED

'''This is intended to help make visualizations for what actually happens, numerically, in these sequences. It is hard to
describe what is going on just by plotting raw values, considering that many different scaler combinations and preprocessing
routines were used. 

Logic flow: (use case of calculating the variation of batch statistics as time goes by)
load sequence (numpy.load) -> instantiate a generator chunker ->  calculate statistics on chunks yielded -> 
 (after each chunk's characteristics have been determined) save calculation results into an array -> plot the array /
  describe the temporary array (== becomes the variation of batch statistics, instead of just variations of batches).'''


# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
chunker_path = analysis_path + "chunker/"
preds_path = analysis_path + "preds/"

save_arrays = True
CHUNKER_BATCH_SIZE = 128  # marked for calculating the endpoint for the generator loading. Sorry about the overloaded/clashing name.
CHUNKER_BATCH_TRAVERSAL_SIZE = 8 #how big of a batch does the chunker yield at each .next() call.

# load the preds
test_seqs_filenames = os.listdir(test_path + 'data/')
test_seqs_filenames.sort()

train_seqs_filenames = os.listdir(train_path + 'data/')
train_seqs_filenames.sort()

test_labels_filenames = os.listdir(test_path + 'label/')
test_labels_filenames.sort()

train_labels_filenames = os.listdir(train_path + 'label/')
train_labels_filenames.sort()

assert len(test_seqs_filenames) == len(test_labels_filenames)
assert len(train_seqs_filenames) == len(train_labels_filenames)

combined_test_filenames = zip(test_seqs_filenames, test_labels_filenames)
combined_train_filenames = zip(train_seqs_filenames, train_labels_filenames)

scalers_list = ['standard','robust','minmax','standard_per_batch','robust_per_batch','minmax_per_batch']
#list of scaler possibilities

batch_stats_dict = {}

#load arrays
#outer for: sequences

#from parser_invidivual_arrays
colnums_translator = ['percent_damage', 'delta_K_current_1', 'ctip_posn_curr_1',
                             'delta_K_current_2',
                             'ctip_posn_curr_2', 'delta_K_current_3', 'ctip_posn_curr_3', 'delta_K_current_4',
                             'ctip_posn_curr_4', 'Load_1', 'Load_2','delta_a_current_1','delta_a_current_2','delta_a_current_3','delta_a_current_4']  # and seq_id,somehow

for index_to_load in range(0, len(combined_test_filenames)):
    files = combined_test_filenames[index_to_load]
    print("files: {}".format(files))
    test_seq_load_path = test_path + 'data/' + files[0]
    test_label_load_path = test_path + 'label/' + files[1]
    test_seq_temp = np.load(test_seq_load_path)
    test_labels = np.load(test_label_load_path)
    print("data/label shape: {}, {}".format(test_seq_temp.shape,test_labels.shape))
    # #create a combined array where seq_temp is appended to the labels.
    combined_shape = (test_seq_temp.shape[0],test_seq_temp.shape[1] + test_labels.shape[1])
    # combined_array = np.zeros(shape=combined_shape)
    # combined_array[:,0:train_seq_temp.shape[1]] = train_seq_temp
    # combined_array[:,train_seq_temp.shape[1]:combined_array.shape[1]] = train_labels

    #print("train array col {}: {}".format(col,describe(train_array[:,col],axis=0)))

    #inner for: scaler types
    for scaler_type_to_test in scalers_list:
        #part with the generator from ensemble conv-lstm
        #load sequence
        #instantiate generator, calculate for how long does the generator have to run

        gen_for_examination = \
            pair_generator_1dconv_lstm_bagged(data=test_seq_temp,labels=test_labels,
                                              start_at = 0, use_precomputed_coeffs=False,scaler_type = scaler_type_to_test,
                                              generator_batch_size = 128,generator_pad = 128,label_dims=4)

        #this is the use case for computing the per-batch statistics
        remaining = CHUNKER_BATCH_SIZE * (test_seq_temp.shape[0] // CHUNKER_BATCH_SIZE)
        generator_yield_total = remaining #number of steps yielded.
        num_cols = test_seq_temp.shape[1] + test_labels.shape[1]#the number of columns in the features array.

        #there will be several final dataframes. Per-batch statistics, per-sequence-statistics.
        #1.
        #THIS ONE IS A TEMP!! will be analyzed.
        batch_stats_array = np.zeros(shape=(generator_yield_total,num_cols*4))
        #to store the flattened per-batch descriptions

        #have a proxy method that determines the batch size before

        temp_array_numrows = test_seq_temp.shape[0] #just the number of rows
        temp_array_numcols = 4 * combined_shape[1] + combined_shape[1]**2 #

        counter = 0
        while remaining > 0:
            counter = counter + 1
            #chunk_stats = describe chunk
            chunk = gen_for_examination.next()
            #need to reshape to a normal array. (get rid of the 1)
            #this is the ENSEMBLE generator's chunk.
            new_chunk_shape = (chunk[0][0].shape[1],chunk[0][0].shape[2] + chunk[1][0].shape[2])

            #chunk to examine SHOULD BE ALL COLUMNS!! #EXAMINE WHY LABEL HAS 5 COLUMNS
            data_chunk_to_examine = np.reshape(chunk[0][0],newshape=(chunk[0][0].shape[1],chunk[0][0].shape[2]))
            #gets rid of the additional axis on the 1st output of the generator's data output
            label_chunk_to_examine = np.reshape(chunk[1][0],newshape=(chunk[1][0].shape[1],chunk[1][0].shape[2]))

            #combined shape explicitly declared; same in the 0th dimension but add the 1st dimensions up.
            combined_chunk_to_examine_shape = (data_chunk_to_examine.shape[0],
                                               data_chunk_to_examine.shape[1] + label_chunk_to_examine.shape[1])
            chunk_to_examine = np.zeros(shape=combined_chunk_to_examine_shape)
            chunk_to_examine[:,0:data_chunk_to_examine.shape[1]] = data_chunk_to_examine
            chunk_to_examine[:,data_chunk_to_examine.shape[1]:] = label_chunk_to_examine

            #get rid of the additional axis on the 1st output of the label part of the generator's output
            # ; the two labels output by the generator are duplicates anyway
            chunk_describe_result = describe(chunk_to_examine,axis=0) #aka the lstm output.

            #chunk_stats_ndarray = dissect chunk_stats so it's saveable as a numpy array
            colnames_list = []
            chunk_mean = chunk_describe_result.mean
            for i in range(0, chunk_mean.shape[0]):
                colnames_list.append('mean_' + colnums_translator[i])

            chunk_variance = chunk_describe_result.variance
            for i in range(0, chunk_mean.shape[0]):
                colnames_list.append('var_' + colnums_translator[i])

            chunk_skewness = chunk_describe_result.skewness
            for i in range(0, chunk_mean.shape[0]):
                colnames_list.append('skew_' + colnums_translator[i])

            chunk_kurtosis = chunk_describe_result.kurtosis
            for i in range(0, chunk_mean.shape[0]):
                colnames_list.append('kur_' + colnums_translator[i])

            chunk_minmax = chunk_describe_result.minmax
            chunk_corrcoef = np.corrcoef(chunk_to_examine,rowvar=False)

            for rownum in range(0,chunk_corrcoef.shape[0]):
                for colnum in range(0,chunk_corrcoef.shape[1]):
                    colnames_list.append('corrcoef_r' + colnums_translator[rownum] + "_c" + colnums_translator[colnum])
            chunk_corrcoef_flattened = chunk_corrcoef.flatten()
            chunk_ndarray = np.zeros(shape=(4 + chunk_to_examine.shape[1],chunk_to_examine.shape[1]))

            chunk_ndarray[0,:] = chunk_mean
            chunk_ndarray[1,:] = chunk_variance
            chunk_ndarray[2,:] = chunk_skewness
            chunk_ndarray[3,:] = chunk_kurtosis
            chunk_ndarray[4:,:] = chunk_corrcoef

            #TODO: add chunk into the batch_stats_array
            #flatten chunk_stats_ndarray
            chunk_ndarray_flattened = np.ndarray.flatten(chunk_ndarray)
            #print("chunk_ndarray_flattened shape: {}".format(chunk_ndarray_flattened.shape))
            remaining = remaining - CHUNKER_BATCH_SIZE
            #define key (sequence name + scaler applied)
        key = files[0] + str(scaler_type_to_test)
        #dict[key] = chunk_stats_ndarray_flattened.
        batch_stats_dict[key] = chunk_ndarray_flattened
        #dict as pandas dataframe.

train_describe_result_df = pd.DataFrame.from_dict(batch_stats_dict, orient='index')
#todo: wrong flatten direction.

print("len(colnames): {} len(flattened): {} colnames_list: {}".format(len(colnames_list),chunk_ndarray_flattened.shape,colnames_list))
#assert(len(colnames_list) == chunk_ndarray_flattened.shape[0])

train_describe_result_df.to_csv(analysis_path + 'test_corpus_describe_result3.csv',header=colnames_list)
# except:
#     print("writing the csv without headers.")
#     train_describe_result_df.to_csv(analysis_path + 'train_corpus_describe_result.csv')
print('csv written.')


#TODO: pca whiten=True.
##################################################### OLD STUFF BELOW
# #load into generator
#
#
# # instantiate variables
# mse_cumulative = 0.0
# mse_at_instance = 0.0
# mse_average = 0.0
# mse_at_instance_list = []
# mse_average_list = []
# mse_cumulative_list = []
#
# mae_cumulative = 0.0
# mae_at_instance = 0.0
# mae_average = 0.0
# mae_at_instance_list = []
# mae_average_list = []
# mae_cumulative_list = []
#
# # med_ae_cumulative = 0.0
# # med_ae_at_instance = 0.0
# # med_ae_average = 0.0
# # med_ae_at_instance_list = []
# # med_ae_average_list = []
# # med_ae_cumulative_list = []
#
# msle_cumulative = 0.0
# msle_at_instance = 0.0
# msle_average = 0.0
# msle_at_instance_list = []
# msle_average_list = []
# msle_cumulative_list = []
#
# evs_cumulative = 0.0
# evs_at_instance = 0.0
# evs_average = 0.0
# evs_at_instance_list = []
# evs_average_list = []
# evs_cumulative_list = []
#
# r2_cumulative = 0.0
# r2_at_instance = 0.0
# r2_average = 0.0
# r2_at_instance_list = []
# r2_average_list = []
# r2_cumulative_list = []
#
# # for index_to_load in range(0,2):
# for index_to_load in range(0, len(test_seqs_filenames)):
#
#     mse_cumulative = 0.0
#     mse_at_instance = 0.0
#     mse_average = 0.0
#     mse_at_instance_list = []
#     mse_average_list = []
#     mse_cumulative_list = []
#
#     mae_cumulative = 0.0
#     mae_at_instance = 0.0
#     mae_average = 0.0
#     mae_at_instance_list = []
#     mae_average_list = []
#     mae_cumulative_list = []
#
#     # med_ae_cumulative = 0.0
#     # med_ae_at_instance = 0.0
#     # med_ae_average = 0.0
#     # med_ae_at_instance_list = []
#     # med_ae_average_list = []
#     # med_ae_cumulative_list = []
#
#     # msle_cumulative = 0.0
#     # msle_at_instance = 0.0
#     # msle_average = 0.0
#     # msle_at_instance_list = []
#     # msle_average_list = []
#     # msle_cumulative_list = []
#
#     evs_cumulative = 0.0
#     evs_at_instance = 0.0
#     evs_average = 0.0
#     evs_at_instance_list = []
#     evs_average_list = []
#     evs_cumulative_list = []
#
#     r2_cumulative = 0.0
#     r2_at_instance = 0.0
#     r2_average = 0.0
#     r2_at_instance_list = []
#     r2_average_list = []
#     r2_cumulative_list = []
#
#     files = combined_filenames[index_to_load]
#     print("files: {}".format(files))
#     preds_load_path = preds_path + files[0]
#     test_label_load_path = test_labels_path + files[1]
#     preds_array_temp = np.load(preds_load_path)
#     label_test_array = np.load(test_label_load_path)
#     print("before changing. preds shape: {}, label shape: {}".format(preds_array_temp.shape, label_test_array.shape))
#     if label_test_array.shape[1] > 5:
#         label_test_array = label_test_array[:, 1:]
#
#     # TODO: reshape the preds.
#     preds_array = np.reshape(preds_array_temp, newshape=(preds_array_temp.shape[1], 4))
#     identifier = files[1][:-4]
#     mse_full = mean_squared_error(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0], 1:])
#     # mse_full_vw = mean_squared_error(y_pred=preds_array,y_true=label_test_array[0:preds_array.shape[0],1:],multioutput='variance_weighted')
#     mae_full = mean_absolute_error(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0], 1:])
#     # mae_full_vw = mean_absolute_error(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0],1:],multioutput='variance_weighted')
#     r2_full = r2_score(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0], 1:])
#     # r2_full_vw = r2_score(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0],1:],multioutput='variance_weighted')
#     evs_full = explained_variance_score(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0], 1:])
#     # evs_full_vw = explained_variance_score(y_pred=preds_array, y_true=label_test_array[0:preds_array.shape[0],1:],multioutput='variance_weighted')
#
#     # if train_array.shape[1] > 11:
#     #     train_array = train_array[:,1:]
#
#     # identifier = files[0][:-4]
#
#     # TODO load predictions
#     # TODO load labels
#     # initialize two sklearn metrics
#
#     # loss_cumulative = loss_temp + mean_squared_error(y_true=label_train_array,y_pred=train_array[:,-4:])
#     # loss_at_instance = mean_squared_error(y_true=None,y_pred=None)
#     # loss_average = loss_cumulative / 5#counter #BASED ON LOSS CUMULATIVE
#     # loss_instance_avg = ?
#
#     chunker_proto_preds = generator_chunker(array_raw=preds_array, chunker_batch_size=CHUNKER_BATCH_TRAVERSAL_SIZE,
#                                             start_at=0,
#                                             scaler_active=False)
#     chunker_proto_label = generator_chunker(array_raw=label_test_array, chunker_batch_size=CHUNKER_BATCH_TRAVERSAL_SIZE,
#                                             start_at=0,
#                                             scaler_active=True, scaler_type='standard_per_batch')
#
#     remaining = CHUNKER_BATCH_SIZE * (preds_array.shape[0] // CHUNKER_BATCH_SIZE)
#     counter = 0
#     # TODO modify this. load both generators and just accumulate the loss.
#     while remaining > 0:
#         counter = counter + 1
#         chunk_preds = chunker_proto_preds.next()
#         # chunk_data = chunk_data[:,-4:] #dummy, just cut the array to the last 4 columns.
#         chunk_label = chunker_proto_label.next()
#         chunk_label = chunk_label[:, 1:]
#
#         # MSE
#         mse_at_instance = mean_squared_error(y_true=chunk_label, y_pred=chunk_preds)
#         mse_at_instance_list.append(mse_at_instance)
#         mse_cumulative = mse_cumulative + mse_at_instance
#         mse_cumulative_list.append(mse_cumulative)
#         mse_average = mse_cumulative / counter
#         mse_average_list.append(mse_average)
#
#         # MAE
#         mae_at_instance = mean_absolute_error(y_true=chunk_label, y_pred=chunk_preds)
#         mae_at_instance_list.append(mae_at_instance)
#         mae_cumulative = mae_cumulative + mae_at_instance
#         mae_cumulative_list.append(mae_cumulative)
#         mae_average = mae_cumulative / counter
#         mae_average_list.append(mae_average)
#
#         # MSLE
#         # msle_at_instance = mean_squared_log_error(y_true=chunk_label,y_pred=chunk_data)
#         # msle_at_instance_list.append(msle_at_instance)
#         # msle_cumulative = msle_cumulative + msle_at_instance
#         # msle_cumulative_list.append(msle_cumulative)
#         # msle_average = msle_cumulative/counter
#         # msle_average_list.append(msle_average)
#
#         # R2
#         r2_at_instance = r2_score(y_true=chunk_label, y_pred=chunk_preds)
#         r2_at_instance_list.append(r2_at_instance)
#         r2_cumulative = r2_cumulative + r2_at_instance
#         r2_cumulative_list.append(r2_cumulative)
#         r2_average = r2_cumulative / counter
#         r2_average_list.append(r2_average)
#
#         # EVS
#         evs_at_instance = explained_variance_score(y_true=chunk_label, y_pred=chunk_preds)
#         evs_at_instance_list.append(evs_at_instance)
#         evs_cumulative = evs_cumulative + evs_at_instance
#         evs_cumulative_list.append(evs_cumulative)
#         evs_average = evs_cumulative / counter
#         evs_average_list.append(evs_average)
#
#         # Med_AE can't do multiple columns at once!
#         # med_ae_at_instance = median_absolute_error(y_true=chunk_label,y_pred=chunk_data) #
#         # med_ae_at_instance_list.append(med_ae_at_instance)
#         # med_ae_cumulative = med_ae_cumulative + med_ae_at_instance
#         # med_ae_cumulative_list.append(med_ae_cumulative)
#         # med_ae_average = med_ae_cumulative/counter
#         # med_ae_average_list.append(med_ae_average)
#
#         print("remaining: {}".format(remaining))
#         remaining = remaining - CHUNKER_BATCH_SIZE
#         # print("data chunk 2: {}".format(chunker_proto_data.next()))
#
#     aggregate_list = []  # saves the aggregated array in a list so the filename saving can be automated
#     aggregate_name_list = []  # since trying to directly access variable names isn't a good idea in python...
#     # MSE MAE MSLE R2 EVS MED_AE
#     print("mse_cumulative: {}".format(mse_cumulative_list))
#     print("mse_average: {}".format(mse_average_list))
#     print("mse_at_instance: {}".format(mse_at_instance_list))
#     assert (len(mse_cumulative_list) == len(mse_average_list) == len(mse_at_instance_list))
#     aggregate_mse = np.empty(shape=(len(mse_cumulative_list), 4))
#     # ORDER IS cumulative - average - at instance
#     aggregate_mse[:, 0] = np.asarray(mse_cumulative_list)
#     aggregate_mse[:, 1] = np.asarray(mse_average_list)
#     aggregate_mse[:, 2] = np.asarray(mse_at_instance_list)
#     aggregate_mse[0, 3] = mse_full
#     # aggregate_mse[1, 3] = mse_full_vw
#     aggregate_list.append(aggregate_mse)
#     aggregate_name_list.append('mse_agg')
#
#     print("mae_cumulative: {}".format(mae_cumulative_list))
#     print("mae_average: {}".format(mae_average_list))
#     print("mae_at_instance: {}".format(mae_at_instance_list))
#     assert (len(mae_cumulative_list) == len(mae_average_list) == len(mae_at_instance_list))
#     aggregate_mae = np.empty(shape=(len(mae_cumulative_list), 4))
#     # ORDER IS cumulative - average - at instance
#     aggregate_mae[:, 0] = np.asarray(mae_cumulative_list)
#     aggregate_mae[:, 1] = np.asarray(mae_average_list)
#     aggregate_mae[:, 2] = np.asarray(mae_at_instance_list)
#     aggregate_mae[0, 3] = mae_full
#     aggregate_mse[1:, 3] = 0.0
#     aggregate_list.append(aggregate_mae)
#     aggregate_name_list.append('mae_agg')
#
#     # print("msle_cumulative: {}".format(msle_cumulative_list))
#     # print("msle_average: {}".format(msle_average_list))
#     # print("msle_at_instance: {}".format(msle_at_instance_list))
#     # assert(len(msle_cumulative_list)==len(msle_average_list)==len(msle_at_instance_list))
#     # aggregate_msle = np.empty(shape=(len(msle_cumulative_list),4))
#     # #ORDER IS cumulative - average - at instance
#     # aggregate_msle[:, 0] = np.asarray(msle_cumulative_list)
#     # aggregate_msle[:, 1] = np.asarray(msle_average_list)
#     # aggregate_msle[:, 2] = np.asarray(msle_at_instance_list)
#     # aggregate_msle[0, 3] = msle_full
#     # aggregate_list.append(aggregate_msle)
#     # aggregate_name_list.append("msle")
#
#     print("r2_cumulative: {}".format(r2_cumulative_list))
#     print("r2_average: {}".format(r2_average_list))
#     print("r2_at_instance: {}".format(r2_at_instance_list))
#     assert (len(r2_cumulative_list) == len(r2_average_list) == len(r2_at_instance_list))
#     aggregate_r2 = np.empty(shape=(len(r2_cumulative_list), 4))
#     # ORDER IS cumulative - average - at instance
#     aggregate_r2[:, 0] = np.asarray(r2_cumulative_list)
#     aggregate_r2[:, 1] = np.asarray(r2_average_list)
#     aggregate_r2[:, 2] = np.asarray(r2_at_instance_list)
#     aggregate_r2[0, 3] = r2_full
#     aggregate_list.append(aggregate_r2)
#     aggregate_name_list.append('r2_agg')
#
#     print("evs_cumulative: {}".format(evs_cumulative_list))
#     print("evs_average: {}".format(evs_average_list))
#     print("evs_at_instance: {}".format(evs_at_instance_list))
#     assert (len(evs_cumulative_list) == len(evs_average_list) == len(evs_at_instance_list))
#     aggregate_evs = np.empty(shape=(len(evs_cumulative_list), 4))
#     # ORDER IS cumulative - average - at instance
#     aggregate_evs[:, 0] = np.asarray(evs_cumulative_list)  # cumulative.
#     aggregate_evs[:, 1] = np.asarray(evs_average_list)
#     aggregate_evs[:, 2] = np.asarray(evs_at_instance_list)
#     aggregate_evs[0, 3] = evs_full
#     aggregate_list.append(aggregate_evs)
#     aggregate_name_list.append('evs_agg')
#
#     if save_arrays == True:
#         for index in range(0, len(aggregate_list)):
#             # get the index of the array in the list of names (the second list)
#             arrayname = aggregate_name_list[index]
#             np.savetxt(fname=chunker_path + arrayname + "_" + str(identifier) + ".csv", delimiter=',',
#                        X=aggregate_list[index], header="cumulative(sum)-average-instance", fmt='%.18e')
#
#
#             # TODO: aggregate and save as a numpy array or a csv.
#
#             # print("med_ae_cumulative: {}".format(med_ae_cumulative_list))
#             # print("med_ae_average: {}".format(med_ae_average_list))
#             # print("med_ae_at_instance: {}".format(med_ae_at_instance_list))
#             # print("label chunk 1: {}".format(chunker_proto_label.next()))
#             # print("label chunk 2: {}".format(chunker_proto_label.next()))
#
#             # if (str(files[0]) == 'sequence_2c_288_9_fv1b.npy') == True:
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:, 0], '^', label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:, 0], '.', label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
#             #     plt.title('truth vs prediction from 75% - 100% of the sequence on Crack 01')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_75_100_newmarker_batch' + str(
#             #         generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:, 1], '^', label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:, 1], 'v', label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
#             #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 02')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_75_100_newmarker_batch' + str(
#             #         generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:, 2], '^', label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:, 2], 'v', label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
#             #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 03')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_75_100_newmarker_batch' + str(
#             #         generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:, 3], '^', label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:, 3], 'v', label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.75 * (len(y_prediction)), 1 * (len(y_prediction))))
#             #     plt.title('truth vs prediction  from 75% - 100% of the sequence on Crack 04')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_75_100_newmarker_batch' + str(
#             #         generator_batch_size) + '_.png')
#             # DEVIN PLOT CODE
#             # if save_figs == True:
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:,0],'^',label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:,0],'.',label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
#             #     plt.title('truth vs prediction from 50% - 100% of the sequence on Crack 01')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_0_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:,1],'^',label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:,1],'v',label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
#             #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 02')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_1_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:,2],'^',label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:,2],'v',label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
#             #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 03')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_2_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
#             #
#             #     plt.clf()
#             #     plt.cla()
#             #     plt.close()
#             #     plt.plot(label_truth[:,3],'^',label="ground truth", markersize=5)
#             #     plt.plot(y_prediction[:,3],'v',label="prediction", markersize=4)
#             #     plt.xscale('log')
#             #     plt.xlabel('# Cycle(s)')
#             #     plt.yscale('log')
#             #     plt.ylabel('Value(s)')
#             #     plt.legend()
#             #     plt.xlim((0.5*(len(y_prediction)), 1*(len(y_prediction))))
#             #     plt.title('truth vs prediction  from 50% - 100% of the sequence on Crack 04')
#             #     plt.grid(True)
#             #     plt.savefig('results_' + str(files[0]) + '_flaw_3_conv_50_100_newmarker_batch' + str(generator_batch_size) + '_.png')
