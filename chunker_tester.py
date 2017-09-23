from corpus_characterizer import generator_chunker
import numpy as np
import os
from sklearn.metrics import mean_squared_error,mean_absolute_error, median_absolute_error, mean_squared_log_error, explained_variance_score, r2_score

#@@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"

#---------part from covariance tester---------------

CHUNKER_BATCH_SIZE = 128
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
#part to determine the total size of the training corpus.
# for index_to_load in range(0,len(combined_filenames)):
#     files = combined_filenames[index_to_load]
#     print("files: {}".format(files))
#     data_load_path = train_path + '/data/' + files[0]
#     label_load_path = train_path + '/label/' + files[1]
#     train_array = np.load(data_load_path)
#     label_train_array = np.load(label_load_path)
#     identifier = files[0][:-4]
#     length_total += train_array.shape[0]
#     #BLOCK FOR AGGREGATED SEQUENCES
# print("total length = {}".format(length_total))
# temp_array = np.empty(shape=(length_total, desired_colnumber))


#instantiate variables
mse_cumulative = 0.0
mse_at_instance = 0.0
mse_average = 0.0
mse_at_instance_list = []
mse_average_list = []
mse_cumulative_list = []

mae_cumulative = 0.0
mae_at_instance = 0.0
mae_average = 0.0
mae_at_instance_list = []
mae_average_list = []
mae_cumulative_list = []

# med_ae_cumulative = 0.0
# med_ae_at_instance = 0.0
# med_ae_average = 0.0
# med_ae_at_instance_list = []
# med_ae_average_list = []
# med_ae_cumulative_list = []

msle_cumulative = 0.0
msle_at_instance = 0.0
msle_average = 0.0
msle_at_instance_list = []
msle_average_list = []
msle_cumulative_list = []

evs_cumulative = 0.0
evs_at_instance = 0.0
evs_average = 0.0
evs_at_instance_list = []
evs_average_list = []
evs_cumulative_list = []

r2_cumulative = 0.0
r2_at_instance = 0.0
r2_average = 0.0
r2_at_instance_list = []
r2_average_list = []
r2_cumulative_list = []

for index_to_load in range(0,2):
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

#TODO load predictions
#TODO load labels
#initialize two sklearn metrics

    # loss_cumulative = loss_temp + mean_squared_error(y_true=label_train_array,y_pred=train_array[:,-4:])
    # loss_at_instance = mean_squared_error(y_true=None,y_pred=None)
    # loss_average = loss_cumulative / 5#counter

    chunker_proto_data = generator_chunker(array_raw=train_array,chunker_batch_size=CHUNKER_BATCH_SIZE,start_at=0)
    chunker_proto_label = generator_chunker(array_raw=label_train_array,chunker_batch_size=CHUNKER_BATCH_SIZE,start_at=0)

    remaining = CHUNKER_BATCH_SIZE*(train_array.shape[0] // CHUNKER_BATCH_SIZE)
    counter = 0
    #TODO modify this. load both generators and just accumulate the loss.
    while remaining > 0:
        counter=counter+1
        chunk_data = chunker_proto_data.next()
        chunk_data = chunk_data[:,-4:] #dummy, just cut the array to the last 4 columns.
        chunk_label = chunker_proto_label.next()
        chunk_label = chunk_label[:,1:]

        #MSE
        mse_at_instance = mean_squared_error(y_true=chunk_label,y_pred=chunk_data)
        mse_at_instance_list.append(mse_at_instance)
        mse_cumulative = mse_cumulative + mse_at_instance
        mse_cumulative_list.append(mse_cumulative)
        mse_average = mse_cumulative/counter
        mse_average_list.append(mse_average)

        #MAE
        mae_at_instance = mean_absolute_error(y_true=chunk_label,y_pred=chunk_data)
        mae_at_instance_list.append(mae_at_instance)
        mae_cumulative = mae_cumulative + mae_at_instance
        mae_cumulative_list.append(mae_cumulative)
        mae_average = mae_cumulative/counter
        mae_average_list.append(mae_average)
        
        #MSLE
        msle_at_instance = mean_squared_log_error(y_true=chunk_label,y_pred=chunk_data)
        msle_at_instance_list.append(msle_at_instance)
        msle_cumulative = msle_cumulative + msle_at_instance
        msle_cumulative_list.append(msle_cumulative)
        msle_average = msle_cumulative/counter
        msle_average_list.append(msle_average)
        
        #R2
        r2_at_instance = r2_score(y_true=chunk_label,y_pred=chunk_data)
        r2_at_instance_list.append(r2_at_instance)
        r2_cumulative = r2_cumulative + r2_at_instance
        r2_cumulative_list.append(r2_cumulative)
        r2_average = r2_cumulative/counter
        r2_average_list.append(r2_average)
        
        #EVS
        evs_at_instance = explained_variance_score(y_true=chunk_label,y_pred=chunk_data)
        evs_at_instance_list.append(evs_at_instance)
        evs_cumulative = evs_cumulative + evs_at_instance
        evs_cumulative_list.append(evs_cumulative)
        evs_average = evs_cumulative/counter
        evs_average_list.append(evs_average)
        
        #Med_AE can't do multiple columns at once!
        # med_ae_at_instance = median_absolute_error(y_true=chunk_label,y_pred=chunk_data) #
        # med_ae_at_instance_list.append(med_ae_at_instance)
        # med_ae_cumulative = med_ae_cumulative + med_ae_at_instance
        # med_ae_cumulative_list.append(med_ae_cumulative)
        # med_ae_average = med_ae_cumulative/counter
        # med_ae_average_list.append(med_ae_average)
        
        print("remaining: {}".format(remaining))
        remaining = remaining - CHUNKER_BATCH_SIZE
        #print("data chunk 2: {}".format(chunker_proto_data.next()))
        
        #MSE MAE MSLE R2 EVS MED_AE
    print("mse_cumulative: {}".format(mse_cumulative_list))
    print("mse_average: {}".format(mse_average_list))
    print("mse_at_instance: {}".format(mse_at_instance_list))

    print("mae_cumulative: {}".format(mae_cumulative_list))
    print("mae_average: {}".format(mae_average_list))
    print("mae_at_instance: {}".format(mae_at_instance_list))
    
    print("msle_cumulative: {}".format(msle_cumulative_list))
    print("msle_average: {}".format(msle_average_list))
    print("msle_at_instance: {}".format(msle_at_instance_list))
    
    print("r2_cumulative: {}".format(r2_cumulative_list))
    print("r2_average: {}".format(r2_average_list))
    print("r2_at_instance: {}".format(r2_at_instance_list))
    
    print("evs_cumulative: {}".format(evs_cumulative_list))
    print("evs_average: {}".format(evs_average_list))
    print("evs_at_instance: {}".format(evs_at_instance_list))

    #TODO: aggregate and save as a numpy array or a csv.

    # print("med_ae_cumulative: {}".format(med_ae_cumulative_list))
    # print("med_ae_average: {}".format(med_ae_average_list))
    # print("med_ae_at_instance: {}".format(med_ae_at_instance_list))
    # print("label chunk 1: {}".format(chunker_proto_label.next()))
    # print("label chunk 2: {}".format(chunker_proto_label.next()))

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

