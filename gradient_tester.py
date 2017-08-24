import numpy as np
import json
import pandas as pd

np.set_printoptions(precision=3,suppress = True, linewidth = 150)
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




#seq_group_params = {}
seq_group_params_filename = "./analysis/seq_group_params.json"
seq_group_params = json.load(open(seq_group_params_filename))

#seq_individual_params = {}
seq_individual_params_filename = "./analysis/seq_individual_params.json"
seq_individual_params = json.load(open(seq_individual_params_filename))


#seq_entire_params = {}
seq_entire_params_filename = "./analysis/seq_entire_params.json"
seq_entire_params = json.load(open(seq_entire_params_filename))

data_array_test_filename = "./train/data/sequence_1b_51_2_fv1b.npy"
label_array_test_filename = "./train/label/sequence_1b_51_2_label_fv1b.npy"
data_array_test = np.load(data_array_test_filename)
label_array_test = np.load(label_array_test_filename)

#onset of linearity is the second derivative. do this numerically.
#stepindex is kept in the label file.
array_to_feed = np.empty(shape=(data_array_test.shape[0],2))
array_to_feed[:,0] = label_array_test[:,0] #stepIndex
array_to_feed[:,1] = np.log(label_array_test[:,1]) #delta_a_current_1
first_derivative_deltaA_flaw0 = np.asarray(np.gradient(array_to_feed,axis=0)) #drop the constant first column.
first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[:,1]

# first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[first_derivative_deltaA_flaw0 == np.inf] = 999
# first_derivative_deltaA_flaw0 = first_derivative_deltaA_flaw0[first_derivative_deltaA_flaw0 == -np.inf] = 0
print(first_derivative_deltaA_flaw0)

array_to_feed[:,1] = first_derivative_deltaA_flaw0
second_derivative_deltaA_flaw0 = np.asarray(np.gradient(array_to_feed,axis=0))
second_derivative_deltaA_flaw0 = second_derivative_deltaA_flaw0[:,1]
print(second_derivative_deltaA_flaw0)
print(np.where(np.diff(np.signbit(second_derivative_deltaA_flaw0)))[0])
array_to_df = np.empty(shape=(data_array_test.shape[0],2))
array_to_df[:,0] = label_array_test[:,0]
array_to_df[:,1] = second_derivative_deltaA_flaw0

second_derivative_df = pd.DataFrame(array_to_df)
#plot = second_derivative_df.iloc[:,1].plot(logx=True)
#plt.savefig('./figure_try.png')

cov_flaw0_flaw1 = np.cov(label_array_test[:,1],label_array_test[:,2], rowvar = False) #each COLUMN is a variable
corrcoef_flaw0_flaw1 = np.cov(label_array_test[:,1],label_array_test[:,2], rowvar = False)
print(cov_flaw0_flaw1)
print(corrcoef_flaw0_flaw1)

relevant_vars_array = np.empty(shape=(data_array_test.shape[0],13))
print("slice of data array shape: {}".format(data_array_test[:,2:].shape))
relevant_vars_array[:,0:9] = data_array_test[:,2:]
print("slice of label array shape: {}".format(label_array_test[:,1:].shape))
relevant_vars_array[:,9:14] = label_array_test[:,1:]

corr_one_sequence = np.cov(data_array_test, rowvar = False) #each COLUMN is a variable
print(corr_one_sequence.shape)
print(corr_one_sequence)

corr_one_sequence_combined = np.cov(relevant_vars_array,rowvar = False)
print(corr_one_sequence_combined.shape)
print(corr_one_sequence_combined)

corrcoef_one_sequence_combined = np.corrcoef(relevant_vars_array,rowvar = False)
print(corrcoef_one_sequence_combined.shape)
print(corrcoef_one_sequence_combined)