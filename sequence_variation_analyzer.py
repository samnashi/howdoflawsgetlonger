from corpus_characterizer import generator_chunker
import numpy as np
import os
# from sklearn.metrics import mean_squared_error,mean_absolute_error, median_absolute_error, mean_squared_log_error, explained_variance_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, \
    r2_score
import pandas as pd

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
chunker_path = analysis_path + "chunker/"
preds_path = analysis_path + "preds/"

save_arrays = True
CHUNKER_BATCH_SIZE = 128  # marked for calculating the endpoint for the generator loading.
CHUNKER_BATCH_TRAVERSAL_SIZE = 8

# load the preds
preds_filenames = os.listdir(preds_path)
preds_filenames.sort()

# for counter in range(0, len(preds_filenames)):
#     print(preds_filenames[counter])

# load the test labels
test_labels_path = test_path + '/label/'
test_label_filenames = os.listdir(test_labels_path)
test_label_filenames.sort()

assert len(test_label_filenames) == len(preds_filenames)
combined_filenames = zip(preds_filenames, test_label_filenames)

#TODO: make this a dataframe.

#df containing variances
#define the coumn names

#df containing correlation
#per scaler method

#df containing dimensionality reduction results of the data
#per scaler method

#df containing scores (prediction losses per batch)
#r2
#MSE
#per scaler method
#per architecture




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
