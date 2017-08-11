import numpy as np
#import matplotlib.pyplot as mpl.pyplot
#import seaborn as sns
import os
from random import shuffle

import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.preprocessing
'''NOT DONE YET!! SCALE THE TRUTH FIRST!!!!!!!!!!! '''
mpl.rcParams['agg.path.chunksize']=100000000000

identifier = "_3_firstrun_fv2_"
Base_Path = "./"
train_path = "/home/ihsan/Documents/thesis_models/train/"
test_path = "/home/ihsan/Documents/thesis_models/test/"
generator_batch_size = 1024

# load data multiple times.
#predictions_filenames = os.listdir(test_path + "predictions")
predictions_filenames = os.listdir('/media/ihsan/LID_FLASH_1/Thesis/Preds_FV1b_Convnet/')
# print("before sorting, data_filenames: {}".format(data_filenames))
predictions_filenames.sort()
# print("after sorting, data_filenames: {}".format(data_filenames))

#label_filenames = os.listdir(test_path + "label")
label_filenames = os.listdir(test_path + "label")
label_filenames.sort()
# print("label_filenames: {}".format(data_filenames))
assert len(predictions_filenames) == len(label_filenames)
combined_filenames = zip(predictions_filenames, label_filenames)
# print("before shuffling: {}".format(combined_filenames))
#shuffle(combined_filenames)
print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.
i=0
#TODO: still only saves single results.
scaler = sklearn.preprocessing.StandardScaler()
for files in combined_filenames:
    i=i+1
    #predictions_load_path = test_path + 'predictions/' + files[0]
    predictions_load_path = '/media/ihsan/LID_FLASH_1/Thesis/Preds_FV1b_Convnet/' + files[0]
    label_load_path = test_path + 'label/' + files[1]
    # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
    y_preds_to_reshape = np.load(predictions_load_path)
    y_truth_unscaled = np.load(label_load_path)[:, 1:]

    print("before reshaping preds/label shape: {}, {}".format(y_preds_to_reshape.shape, y_truth_unscaled.shape))
    y_preds = np.reshape(y_preds_to_reshape,newshape=(y_preds_to_reshape.shape[1],y_preds_to_reshape.shape[2]))
    #y_truth_unscaled = np.reshape(y_truth_unscaled,newshape=(y_truth_unscaled.shape[0],y_truth_unscaled.shape[1]))
    y_truth = scaler.fit_transform(X=y_truth_unscaled, y=None)
    predictions_length = generator_batch_size * (y_truth.shape[0] // generator_batch_size)
    print("before resampling preds/label shape: {}, {}".format(y_preds.shape, y_truth.shape))
    #--------COMMENTED OUT BECAUSE OF SCALER IN THE GENERATOR-----------------------------------
    #test_array = np.reshape(test_array, (1, test_array.shape[0], test_array.shape[1]))
    #label_array[0:,0:predictions_length,:] = np.reshape(label_array[0:,0:predictions_length,:],(1,label_array[0:,0:predictions_length,:].shape[0],label_array[0:,0:predictions_length,:].shape[1])) #label doesn't need to be 3D
    # cut the label to be the same length as the predictions.
    y_truth = y_truth[0:y_preds.shape[0],:]
    #intervals = np.linspace(start=0, stop=y_preds.shape[0],num=2000)
    resample_interval = 8
    axis_option = 'log'
    y_preds=y_preds[::resample_interval,:]
    y_truth = y_truth[::resample_interval,:]
    print("filename: {}, preds/label shape: {}, {}".format(str(files[0]),y_preds.shape, y_truth.shape))
    #predictions_length = generator_batch_size * (y_truth.shape[0]//generator_batch_size)
    #largest integer multiple of the generator batch size that fits into the length of the sequence.

    #print("array to print's shape: {}".format(y_preds[int(0.75*y_preds.shape[0]):,:].shape))
    print("array to print's shape: {}".format(y_preds.shape))
    #np.save(file=('predictions_lstm_'+str(files[0])),arr=y_truth[0:,0:predictions_length,:])
    #x_range= np.arange(start=0, stop=label_array[0:,0:predictions_length,:].shape[1])
    # mpl.pyplot.scatter(x=x_range,y=y_pred[0,:,0])
    # mpl.pyplot.scatter(x=x_range,y=label_array[0:,0:predictions_length,:][0,:,0])
    mpl.pyplot.cla()
    mpl.pyplot.clf()
    mpl.pyplot.close()
    # mpl.pyplot.plot(y_preds[0, int(0.95*predictions_length):, 0],'o')
    # mpl.pyplot.plot(y_truth[0, int(0.95*predictions_length):, 0],'^')
    # mpl.pyplot.plot(y_preds[0, int(0.95*predictions_length):, 1],'o')
    # mpl.pyplot.plot(y_truth[0, int(0.95*predictions_length):, 1],'^')
    #mpl.pyplot.scatter(x=y_preds[:, 0],y=y_truth[:, 0])
    mpl.pyplot.plot(y_preds[0.75*y_preds.shape[0]:, 0],"o")
    mpl.pyplot.plot(y_truth[0.75*y_truth.shape[0]:, 0],"^")
    mpl.pyplot.yscale(axis_option)
    mpl.pyplot.xscale('log')
    mpl.pyplot.title('pred vs. y_truth')
    mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    mpl.pyplot.xlabel('cycles * ' + str(resample_interval))
    #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    mpl.pyplot.legend(['pred[0]', 'true[0]'], loc='upper left')
    #mpl.pyplot.show()
    mpl.pyplot.savefig(test_path + str(files[0])[:-4] + '_conv_detail_results_flaw_0' + '.png', bbox_inches='tight')
    mpl.pyplot.cla()
    mpl.pyplot.clf()
    mpl.pyplot.close()
    mpl.pyplot.plot(y_preds[0.75*y_preds.shape[0]:, 1],"o")
    mpl.pyplot.plot(y_truth[0.75*y_truth.shape[0]:, 1],"^")
    mpl.pyplot.yscale(axis_option)
    mpl.pyplot.xscale('log')
    mpl.pyplot.title('pred vs. y_truth')
    mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    mpl.pyplot.xlabel('cycles * ' + str(resample_interval))
    #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    mpl.pyplot.legend(['pred[1]', 'true[1]'], loc='upper left')
    #mpl.pyplot.show()
    mpl.pyplot.savefig(test_path + str(files[0])[:-4] + '_conv_detail_results_flaw_1' + '.png', bbox_inches='tight')
    mpl.pyplot.cla()
    mpl.pyplot.clf()
    mpl.pyplot.close()

    mpl.pyplot.plot(y_preds[0.75*y_preds.shape[0]:, 2],"o")
    mpl.pyplot.plot(y_truth[0.75*y_truth.shape[0]:, 2],"^")
    mpl.pyplot.yscale(axis_option)
    mpl.pyplot.xscale('log')
    mpl.pyplot.title('pred vs. y_truth')
    mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    mpl.pyplot.xlabel('cycles * ' + str(resample_interval))
    #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    mpl.pyplot.legend(['pred[2]', 'true[2]'], loc='upper left')
    #mpl.pyplot.show()
    mpl.pyplot.savefig(test_path + str(files[0])[:-4] + '_conv_detail_results_flaw_2' + '.png', bbox_inches='tight')
    mpl.pyplot.cla()
    mpl.pyplot.clf()
    mpl.pyplot.close()

    mpl.pyplot.plot(y_preds[0.75*y_preds.shape[0]:, 3],"o")
    mpl.pyplot.plot(y_truth[0.75*y_truth.shape[0]:, 3],"^")
    mpl.pyplot.yscale(axis_option)
    mpl.pyplot.xscale('log')
    mpl.pyplot.title('pred vs. y_truth')
    mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    mpl.pyplot.xlabel('cycles * ' + str(resample_interval))
    #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    mpl.pyplot.legend(['pred[3]', 'true[3]'], loc='upper left')
    #mpl.pyplot.show()
    mpl.pyplot.savefig(test_path + str(files[0])[:-4] + '_conv_detail_results_flaw_3' + '.png', bbox_inches='tight')
    mpl.pyplot.cla()
    mpl.pyplot.clf()
    mpl.pyplot.close()
    #
    # # mpl.pyplot.scatter(x= x_range,y=y_pred[0, :, 2])
    # # mpl.pyplot.scatter(x=x_range, y=label_array[0:,0:predictions_length,:][0, :, 2])
    # mpl.pyplot.plot(y_pred[0, :, 2])
    # mpl.pyplot.plot(label_array[0:,0:predictions_length,:][0, :, 2])
    # mpl.pyplot.yscale((axis_option))
    # mpl.pyplot.xscale((axis_option))
    # mpl.pyplot.title('pred vs. label_array[0:,0:predictions_length,:]')
    # mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    # mpl.pyplot.xlabel('epoch')
    # #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    # mpl.pyplot.legend(['pred[1]', 'true[1]'], loc='upper left')
    #
    # mpl.pyplot.savefig(str(files[0]) + '_results_flaw_2' + '.png', bbox_inches='tight')
    # mpl.pyplot.clf()
    # mpl.pyplot.cla()
    # mpl.pyplot.close()
    #
    # # mpl.pyplot.scatter(x= x_range,y=y_pred[0, :, 3])
    # # mpl.pyplot.scatter(x=x_range,y=label_array[0:,0:predictions_length,:][0, :, 3])
    # mpl.pyplot.plot(y_pred[0, :, 3])
    # mpl.pyplot.plot(label_array[0:,0:predictions_length,:][0, :, 3])
    # mpl.pyplot.yscale((axis_option))
    # mpl.pyplot.xscale((axis_option))
    # mpl.pyplot.title('pred vs. label_array[0:,0:predictions_length,:]')
    # mpl.pyplot.ylabel('crack growth rate, normalized and centered, in/cycle')
    # mpl.pyplot.xlabel('epoch')
    # #mpl.pyplot.legend(['pred[0]', 'true[0]','pred[1]', 'true[1]','pred[2]', 'true[2]','pred[3]','true[3]'], loc='upper left')
    # mpl.pyplot.legend(['pred[1]', 'true[1]'], loc='upper left')
    #
    # mpl.pyplot.savefig(str(files[0]) + '_results_flaw_3' + '.png', bbox_inches='tight')
    # mpl.pyplot.clf()
    # mpl.pyplot.cla()
    # mpl.pyplot.close()
        #print("Score: {}".format(score)) #test_array.shape[0]//generator_batch_size
    # #predictions = model.predict_generator(test_generator, steps=(1*test_array.shape[0]//generator_batch_size),max_queue_size=test_array.shape[0],use_multiprocessing=True)
    # print("scores: {}".format(score))
    # np.savetxt(Base_Path + 'results/TestResult_' + str(num_sequence_draws) + identifier + '.txt', np.asarray(score),
    #            fmt='%5.6f', delimiter=' ', newline='\n', header='loss, acc',
    #            footer=str(), comments='# ')




