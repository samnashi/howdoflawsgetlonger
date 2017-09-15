import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from itertools import combinations_with_replacement

#modifying the scattergro parser so that it reads numpy arrays (from the training corpus) instead.

def analyze_corpus(save_arrays=True, analysis_mode=False, feature_identifier='FVX', use_data_in_model_folder=False):
    '''Cuts apart the lumped arrays. If analysis_mode is True, it'll return the intermediate arrays and indices,
     for corpus_characterizer to do its thing. '''
    # raw_path = "/home/ihsan/Documents/thesis_generator/results/devin/to_process/" #needs the absolute path, no tildes!
    # processed_path = "/home/ihsan/Documents/thesis_generator/results/devin"

    # usb drive
    # raw_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/'

    # --------PASTED PART

    # ------------------------------------------ END OF PASTED PART----------------------------------------

    data_path = "/home/ihsan/Documents/thesis_models/train/data/"
    label_path = "/home/ihsan/Documents/thesis_models/train/label/"

    processed_path = "/home/ihsan/Documents/thesis_models/analysis/"
    # processed_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/processed/'
    items = os.listdir(data_path)
    items.sort()
    print(type(items))
    for file in items:
        if ('.npy') not in str(file):
            del items[items.index(file)]
    print(items)

    seq_length_dict = {}
    seq_length_dict_filename = processed_path + "/sequence_lengths.json"

    seq_group_params = {}
    seq_group_params_filename = "./analysis/seq_group_params.json"

    seq_individual_params = {}
    seq_individual_params_filename = "./analysis/seq_individual_params.json"

    seq_entire_params = {}
    seq_entire_params_filename = "./analysis/seq_entire_params.json"

    j = 0  # counter.
    threshold = 0.5  # threshold crack length. rivet pitch is 0.875" so a bit over half of that.

    for file in items:
        print("filename: {}".format(str(file)))
        npy_path = data_path + str(file)
        if ("_0.") in str(file):  # only the first file in the series has a header.
            train_array = np.load(npy_path)
            label_array = np.load(npy_path) #label.. #TODO MAKE THIS A PROPER LABEL
            header_names = cg_seq_df.columns.values
            print("header names: {}".format(header_names))
        else:
            train_array = pd.read_csv(npy_path, names=header_names)
            print(cg_seq_df.columns.values)

        #TODO: modify train_df. make that a plain old numpy array (since I need the numpy function to get a covariance matrix..)
        print("cg_seq_df shape: {}".format(cg_seq_df.columns.values))
        print(train_df.index, train_df.head(1))
        label_train_df = cg_seq_df[label_list]
        # to accommodate different feature sets, read the column names on the fly.
        seq_group_params['train_colnames'] = train_df.columns.tolist()
        seq_group_params['label_colnames'] = label_train_df.columns.tolist()

        # ------------ANALYSIS PART-----------------------------------------------------------------------------
        if analysis_mode == True:
            # calculates the characteristic parameters of blocks of sequences (same IC and same load cond)
            group_train_scaler_params = {}
            group_label_scaler_params = {}

            group_train_scaler = StandardScaler()
            group_label_scaler = StandardScaler()

            group_train_scaler.fit(train_df.values)
            group_label_scaler.fit(label_train_df.values)
            # print(group_train_scaler.mean_, group_train_scaler.scale_, group_train_scaler.var_, group_train_scaler.std_)

            group_train_scaler_params['mean'] = np.ndarray.tolist(group_train_scaler.mean_)
            group_train_scaler_params['scale'] = np.ndarray.tolist(group_train_scaler.scale_)
            # group_train_scaler_params['std'] = np.ndarray.tolist(group_train_scaler.std_)
            group_train_scaler_params['var'] = np.ndarray.tolist(group_train_scaler.var_)

            group_label_scaler_params['mean'] = np.ndarray.tolist(group_label_scaler.mean_)
            group_label_scaler_params['scale'] = np.ndarray.tolist(group_label_scaler.scale_)
            # group_label_scaler_params['std'] = np.ndarray.tolist(group_label_scaler.std_)
            group_label_scaler_params['var'] = np.ndarray.tolist(group_label_scaler.var_)

            # nested dict.
            seq_group_params[str(file)] = {}
            seq_group_params[str(file)]["data"] = group_train_scaler_params
            seq_group_params[str(file)]["label"] = group_label_scaler_params
        # ------------END OF ANALYSIS PART---------------------------------------------------------------------
        indices = train_df[train_df['StepIndex'] == 1].index.tolist()
        indices.append(train_df.shape[0] - 1)  # the 0th position was missing if run using the original method.
        indices_offset_min1 = [i - 1 for i in indices]
        print("file {}'s indices_offset_min1 {}".format(str(file), indices_offset_min1))
        indices_offset_min1.pop(0)
        print("indices: {}, indices_offset_min1: {}".format(indices, indices_offset_min1))

        ranges = [(t, s) for t, s in zip(indices, indices_offset_min1)]
        # print("before changing :{}".format(ranges))
        '''for tuple in ranges:
            print(tuple)
            tuple[1:][0] = (tuple[1:][0]) + 1'''

        # ranges[1:][0] = ranges[1:][0] + 1
        print("\nafter changing :{} ".format(ranges))
        # print("lengths: {} ".format([indices[4]-indices[3],indices[3]-indices[2],indices[2]-indices[1],indices[1]-indices[0]]))
        print("lengths: {} ".format([t - s for (s, t) in ranges]))

        i = 0

        for indices_as_tuples in ranges:
            i = i + 1
            print("indices as tuples: {}".format(indices_as_tuples))
            train_df_as_np_array = train_df[indices_as_tuples[0]:indices_as_tuples[1]].values
            label_train_df_as_np_array = label_train_df[indices_as_tuples[0]:indices_as_tuples[1]].values
            print("df_as_np_array shape: {}".format(train_df_as_np_array.shape))
            print("file: {}".format(file))
            # TODO makes this a regex.. second underscore
            identifier = str(str(file)[-8:-6])  # eg 1a 2a etc. #you can use a regex.
            print("identifier: {}".format(identifier))

            # TODO feature version identifier.
            # j is sequence id. #i is the sequence number within the csv.
            np_train_path = processed_path + "/sequence_" + identifier + "_" + str(j) + "_" + str(i) + ".npy"
            np_label_train_path = processed_path + "/sequence_" + identifier + "_" + str(j) + "_" + str(
                i) + "_label_.npy"
            seq_length_dict["sequence_" + identifier + "_" + str(j) + "_" + str(i)] = indices_as_tuples[1] - \
                                                                                      indices_as_tuples[0]
            # seq_length_dict = json.load(open(seq_length_dict))
            print("np_train_path: {}".format(np_train_path))
            print("np_label_train_path :{}".format(np_label_train_path))
            # ------------ANALYSIS PART-----------------------------------------------------------------------------
            if analysis_mode == True:  # calculates statistics
                # calculates the characteristic parameters of blocks of sequences (same IC and same load cond)
                individual_sequence_scaler_params = {}
                individual_label_scaler_params = {}

                individual_sequence_scaler = StandardScaler()
                individual_label_scaler = StandardScaler()

                individual_sequence_scaler.fit(train_df_as_np_array)
                individual_label_scaler.fit(label_train_df_as_np_array)
                # print(individual_sequence_scaler.mean_, individual_sequence_scaler.scale_, individual_sequence_scaler.var_, individual_sequence_scaler.std_)

                individual_sequence_scaler_params['mean'] = np.ndarray.tolist(individual_sequence_scaler.mean_)
                individual_sequence_scaler_params['scale'] = np.ndarray.tolist(individual_sequence_scaler.scale_)
                # individual_sequence_scaler_params['std'] = np.ndarray.tolist(individual_sequence_scaler.std_)
                individual_sequence_scaler_params['var'] = np.ndarray.tolist(individual_sequence_scaler.var_)

                individual_label_scaler_params['mean'] = np.ndarray.tolist(individual_label_scaler.mean_)
                individual_label_scaler_params['scale'] = np.ndarray.tolist(individual_label_scaler.scale_)
                # deprecated individual_label_scaler_params['std'] = np.ndarray.tolist(individual_label_scaler.std_)
                individual_label_scaler_params['var'] = np.ndarray.tolist(individual_label_scaler.var_)

                # nested dict.
                seq_individual_params[
                    "sequence_" + identifier + "_" + str(j) + "_" + str(i) + ".npy"] = individual_sequence_scaler_params
                seq_individual_params["sequence_" + identifier + "_" + str(j) + "_" + str(
                    i) + "_label_.npy"] = individual_label_scaler_params
            # ------------END OF ANALYSIS PART----------------------------------------------------------------------
            if save_arrays == True:
                np.save(np_train_path, train_df_as_np_array)
                np.save(np_label_train_path, label_train_df_as_np_array)
            j = j + 1

        print(seq_length_dict)  # these are of individual sequence lengths.
        # ---------------ANALYSIS OF UNSPLIT---------------------------------------------------------------------
        if analysis_mode == True:
            # processed_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/processed/'
            items_processed = os.listdir(processed_path)
            items_processed.sort()
            print(type(items_processed))
            for file_p in items_processed:
                if ('.npy') not in str(file_p):
                    del items_processed[items_processed.index(file_p)]  # get rid of non .npy files from this list.
            print(items_processed)

            # run standardscaler on all the sequences. Would be unproductive to do it earlier.
            entire_data_scaler = StandardScaler()
            entire_label_scaler = StandardScaler()

            entire_data_scaler_params = {}
            entire_label_scaler_params = {}

            for file_p in items_processed:  # TODO these are all tuples..
                if ("label") in str(file_p):
                    partial_label = np.load(processed_path + '/' + str(file_p))
                    entire_label_scaler.partial_fit(partial_label)
                if ("label") not in str(file_p):
                    partial_data = np.load(processed_path + '/' + str(file_p))
                    entire_data_scaler.partial_fit(partial_data)

            entire_data_scaler_params['mean'] = np.ndarray.tolist(entire_data_scaler.mean_)
            entire_data_scaler_params['scale'] = np.ndarray.tolist(entire_data_scaler.scale_)
            # entire_data_scaler_params['std'] = np.ndarray.tolist(entire_data_scaler.std_)
            entire_data_scaler_params['var'] = np.ndarray.tolist(entire_data_scaler.var_)

            entire_label_scaler_params['mean'] = np.ndarray.tolist(entire_label_scaler.mean_)
            entire_label_scaler_params['scale'] = np.ndarray.tolist(entire_label_scaler.scale_)
            # entire_label_scaler_params['std'] = np.ndarray.tolist(entire_label_scaler.std_)
            entire_label_scaler_params['var'] = np.ndarray.tolist(entire_label_scaler.var_)
            seq_entire_params['data'] = entire_data_scaler_params
            seq_entire_params['label'] = entire_label_scaler_params

            # TODO calculate covariances of everything.
            # possible_combinations = combinations_with_replacement(#column numbers ,r=2)
            # crack position vs crack growth rate
            # load vs. crack growth rate

            # TODO find the kink in crack growth rate.
            # probably the correlation between the load and the crack growth rate, on each crack..
            # use pearson_r

        # ---------------END OF ANALYSIS---------------------------------------------------------------------
        # TODO use DictWriter to get csvs.
        json.dump(seq_length_dict, open(seq_length_dict_filename, 'wb'))
        json.dump(seq_group_params, open(seq_group_params_filename, 'wb'))
        json.dump(seq_individual_params, open(seq_individual_params_filename, 'wb'))
        json.dump(seq_entire_params, open(seq_entire_params_filename, 'wb'))


if __name__ == "__main__":
    parse_scattergro(save_arrays=False, analysis_mode=True)

