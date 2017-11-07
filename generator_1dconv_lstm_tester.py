from Conv1D_ActivationSearch_BigLoop import pair_generator_1dconv_lstm
import os
from random import shuffle
import pandas as pd
import numpy as np

# train_generator = pair_generator_1dconv_lstm(train_array, label_array, start_at=active_starting_position,
#                                              generator_batch_size=generator_batch_size,
#                                              use_precomputed_coeffs=use_precomp_sscaler, scaled=scaler_active,
#                                              scaler_type=active_scaler_type)

# @@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if __name__ == "__main__":

    Base_Path = "./"
    image_path = "./images/"
    train_path = "./train/"
    test_path = "./test/"
    analysis_path = "./analysis/"

    scaler_active = True
    active_scaler_type = 'minmax_per_batch' #'standard_per_batch' 'robust_per_batch'
    use_precomp_sscaler = False
    generator_batch_size =128
    active_starting_position=0

    num_sequence_draws = 1
    # load data multiple times.
    data_filenames = os.listdir(train_path + "data")
    # print("before sorting, data_filenames: {}".format(data_filenames))
    data_filenames.sort()
    # print("after sorting, data_filenames: {}".format(data_filenames))

    label_filenames = os.listdir(train_path + "label")
    label_filenames.sort()  # sorting makes sure the label and the data are lined up.
    # print("label_filenames: {}".format(data_filenames))
    assert len(data_filenames) == len(label_filenames)
    combined_filenames = zip(data_filenames, label_filenames)
    # print("before shuffling: {}".format(combined_filenames))
    shuffle(combined_filenames)
    print("after shuffling: {}".format(combined_filenames))  # shuffling works ok.
    print('Data loaded.')
    weights_file_name = None

    for i in range(0, num_sequence_draws):
        index_to_load = np.random.randint(0, len(combined_filenames))  # switch to iterations
        files = combined_filenames[index_to_load]
        print("files: {}".format(files))
        data_load_path = train_path + '/data/' + files[0]
        label_load_path = train_path + '/label/' + files[1]
        # print("data/label load path: {} \n {}".format(data_load_path,label_load_path))
        train_array = np.load(data_load_path)
        label_array = np.load(label_load_path)[:, 1:]
        if train_array.shape[1] != 11:
            train_array = train_array[:, 1:]
        print("data/label shape: {}, {}, draw #: {}".format(train_array.shape, label_array.shape, i))
        # train_array = np.reshape(train_array,(1,generator_batch_size,train_array.shape[1]))
        # label_array = np.reshape(label_array,(1,label_array.shape[0],label_array.shape[1])) #label needs to be 3D for TD!


        nonlinear_part_starting_position = generator_batch_size * ((train_array.shape[0] // generator_batch_size) - 3)
        shuffled_starting_position = np.random.randint(0, nonlinear_part_starting_position)
        active_starting_position = shuffled_starting_position  # doesn't start from 0, if the model is still in the 1st phase of training

        gen_conv1d_lstm_test = pair_generator_1dconv_lstm(train_array, label_array, start_at=active_starting_position,
                                                 generator_batch_size=generator_batch_size,
                                                 use_precomputed_coeffs=use_precomp_sscaler, scaled=scaler_active,
                                                 scaler_type=active_scaler_type)
        for i in range(0, (train_array.shape[0] // generator_batch_size)):
            #batch_to_analyze = np.ndarray(shape=())
            output_unsliced = gen_conv1d_lstm_test.next()
            output_0_all = np.asarray(output_unsliced[0][0])
            output_1_all = np.asarray(output_unsliced[1][0])
            print(i, output_0_all.shape,output_1_all.shape) #(1,256,1)
            #try to cast as an array
            #use scipy describe
            #set equal to
        #print(gen_conv1d_lstm_test.next)