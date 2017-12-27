from scipy.stats import describe, kurtosistest, skewtest, normaltest
from corpus_characterizer import generator_chunker
import numpy as np
import os
import pandas as pd
from AuxRegressor import create_testing_set, create_training_set
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

#for index_to_load in range(0,len(combined_filenames)):
data_rows_list = []
label_rows_list = []
for index_to_load in range(0,len(combined_filenames)):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + 'data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    if train_array.shape[1] > 11:
        train_array = train_array[:,1:]
    if label_train_array.shape[1] >= 5:
        label_train_array = label_train_array[:,1:]
    identifier = files[0][:-4]
    results_dict = {}
    results_label_dict = {}
    results_dict['seq_name'] = identifier
    results_label_dict['seq_name'] = identifier
    for col in range (0,train_array.shape[1]):
        description = describe(train_array[:,col],axis=0)
        kurt = kurtosistest(train_array[:, col], axis=0)
        skew = skewtest(train_array[:, col], axis=0)
        normality = normaltest(train_array[:, col], axis=0)
        results_dict['col_'+str(col) + '_min'] = description.minmax[0]
        results_dict['col_'+str(col) + '_max'] = description.minmax[1]
        #results_dict['col_'+str(col) + '_var'] = description.variance
        results_dict['col_'+str(col) + '_normality'] = normality.statistic
        results_dict['col_' + str(col) + '_skewness'] = skew.statistic
        results_dict['col_' + str(col) + '_kurtosis'] = kurt.statistic
        print("train array col {}: {}".format(col,description))
        print("kurtosis {}".format(kurt))
        print("skew {}".format(skew))
        print("normal {}".format(normality))
    data_rows_list.append(results_dict)
    for col in range (0,label_train_array.shape[1]):
        description = describe(label_train_array[:,col],axis=0)
        kurt = kurtosistest(label_train_array[:, col], axis=0)
        skew = skewtest(label_train_array[:, col], axis=0)
        normality = normaltest(label_train_array[:, col], axis=0)
        results_label_dict['col_'+str(col) + '_min'] = description.minmax[0]
        results_label_dict['col_'+str(col) + '_max'] = description.minmax[1]
        #results_label_dict['col_'+str(col) + '_var'] = description.variance
        results_label_dict['col_'+str(col) + '_normality'] = normality.statistic
        results_label_dict['col_' + str(col) + '_skewness'] = skew.statistic
        results_label_dict['col_' + str(col) + '_kurtosis'] = kurt.statistic
        print("train array col {}: {}".format(col,description))
        print("kurtosis {}".format(kurt))
        print("skew {}".format(skew))
        print("normal {}".format(normality))
    label_rows_list.append(results_label_dict)
print("ha")
labels_stats_df = pd.DataFrame(data=label_rows_list, columns=label_rows_list[0].keys())
labels_stats_df.set_index(['seq_name'])
#labels_stats_df.columns = sorted(list(labels_stats_df.columns).sort())

data_stats_df = pd.DataFrame(data=data_rows_list,columns=data_rows_list[0].keys())
data_stats_df.set_index(['seq_name'])
#data_stats_df.columns = sorted(list(data_stats_df.columns))

labels_stats_df.to_csv(analysis_path + 'label_stats.csv')
data_stats_df.to_csv(analysis_path + 'data_stats.csv')
        #TODO: cast each column into a dataframe row.

    