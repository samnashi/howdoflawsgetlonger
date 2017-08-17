import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

'''train_df_play['percent_damage'] = \
    train_df_play[['crack_length_1','crack_length_2','crack_length_3','crack_length_4']].max(axis=1)/threshold * 100
print(train_df_play.head(2),train_df_play.index,train_df_play.info())

train_df_play = train_df_play[['StepIndex', 'percent_damage','delta_K_current_1','crack_length_1','delta_K_current_2',
 'crack_length_2','delta_K_current_3','crack_length_3','delta_K_current_4',
                              'crack_length_4','Load_1','Load_2']]
print("after changing: {}".format(train_df_play.columns.values))
train_df_play.to_csv('/home/ihsan/Documents/thesis_models/with_stepindex.csv')
#-----------------------SEE BELOW FOR REINDEXING EXAMPLE -------------------------------
train_df_play_dropped_stepindex = train_df_play[['percent_damage','delta_K_current_1','crack_length_1','delta_K_current_2',
 'crack_length_2','delta_K_current_3','crack_length_3','delta_K_current_4',
                              'crack_length_4','Load_1','Load_2']]'''

def parse_scattergro(save_arrays = True, analysis_mode = False, feature_identifier = 'FVX', use_data_in_model_folder = False):
    '''Cuts apart the lumped arrays. If analysis_mode is True, it'll return the intermediate arrays and indices,
     for corpus_characterizer to do its thing. '''
    #raw_path = "/home/ihsan/Documents/thesis_generator/results/devin/to_process/" #needs the absolute path, no tildes!
    #processed_path = "/home/ihsan/Documents/thesis_generator/results/devin"

    #usb drive
    #raw_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/'

    #--------PASTED PART

    #------------------------------------------ END OF PASTED PART----------------------------------------

    raw_path = "/home/ihsan/Documents/thesis_generator/results/to_process/"

    processed_path = "/home/ihsan/Documents/thesis_models/unsplit"
    #processed_path = '/media/ihsan/LID_FLASH_1/Thesis/thesis_generator/results/run_2/processed/'
    items = os.listdir(raw_path)
    items.sort()
    print(type(items))
    for file in items:
        if ('.csv') not in str(file):
            del items[items.index(file)]
    print(items)

    seq_length_dict = {}
    seq_length_dict_filename = processed_path + "/sequence_lengths.json"

    seq_group_params = {}
    seq_group_params_filename = "./analysis/seq_group_params.json"
    
    seq_individual_params = {}
    seq_individual_params_filename = "./analysis/seq_individual_params.json"

    #suffix = "3a"
    #csv_path = "~/Documents/thesis_generator/results/devin/crack_growth_sequence" + suffix + ".csv"
    sequence_lengths = {} #save sequence lengths as a dict. or maybe a json?

    j = 0  # counter.
    threshold = 0.5  # threshold crack length. rivet pitch is 0.875" so a bit over half of that.

    for file in items:
        print("filename: {}".format(str(file)))
        csv_path = raw_path + str(file)
        if ("_0.") in str(file):  # only the first file in the series has a header.
            cg_seq_df = pd.read_csv(csv_path)
            header_names = cg_seq_df.columns.values
            print("header names: {}".format(header_names))
        else:
            cg_seq_df = pd.read_csv(csv_path, names=header_names)
            print(cg_seq_df.columns.values)

        cg_seq_df['percent_damage'] = \
            cg_seq_df[['crack_length_1', 'crack_length_2', 'crack_length_3', 'crack_length_4']].max(
                axis=1) / threshold * 100
        train_list = ['StepIndex', 'percent_damage', 'delta_K_current_1', 'ctip_posn_curr_1', 'delta_K_current_2',
                      'ctip_posn_curr_2',
                      'delta_K_current_3', 'ctip_posn_curr_3', 'delta_K_current_4', 'ctip_posn_curr_4', 'Load_1',
                      'Load_2']  # and seq_id,somehow

        label_list = ['StepIndex', 'delta_a_current_1', 'delta_a_current_2', 'delta_a_current_3', 'delta_a_current_4']

        train_df = cg_seq_df[train_list]
        print("cg_seq_df shape: {}".format(cg_seq_df.columns.values))
        print(train_df.index, train_df.head(1))
        label_train_df = cg_seq_df[label_list]
        # to accommodate different feature sets, read the column names on the fly.
        seq_group_params['train_colnames'] = train_df.columns.tolist()
        seq_group_params['label_colnames'] = label_train_df.columns.tolist()

#------------ANALYSIS PART-----------------------------------------------------------------------------
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
            #group_train_scaler_params['std'] = np.ndarray.tolist(group_train_scaler.std_)
            group_train_scaler_params['var'] = np.ndarray.tolist(group_train_scaler.var_)

            group_label_scaler_params['mean'] = np.ndarray.tolist(group_label_scaler.mean_)
            group_label_scaler_params['scale'] = np.ndarray.tolist(group_label_scaler.scale_)
            #group_label_scaler_params['std'] = np.ndarray.tolist(group_label_scaler.std_)
            group_label_scaler_params['var'] = np.ndarray.tolist(group_label_scaler.var_)

            # nested dict.
            seq_group_params[str(file)] = {}
            seq_group_params[str(file)]["train"] = group_train_scaler_params
            seq_group_params[str(file)]["label"] = group_label_scaler_params
# ------------END OF ANALYSIS PART---------------------------------------------------------------------
        indices = train_df[train_df['StepIndex'] == 1].index.tolist()
        indices.append(train_df.shape[0] - 1) #the 0th position was missing if run using the original method. 
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
            #TODO makes this a regex.. second underscore
            identifier = str(str(file)[-8:-6])  # eg 1a 2a etc. #you can use a regex.
            print("identifier: {}".format(identifier))

            #TODO feature version identifier.
            # j is sequence id. #i is the sequence number within the csv.
            np_train_path = processed_path + "/sequence_" + identifier + "_" + str(j) + "_" + str(i) + ".npy"
            np_label_train_path = processed_path + "/sequence_" + identifier + "_" + str(j) + "_" + str(i) + "_label_.npy"
            seq_length_dict["sequence_" + identifier + "_" + str(j) + "_" + str(i)] = indices_as_tuples[1] - \
                                                                                      indices_as_tuples[0]
            # seq_length_dict = json.load(open(seq_length_dict))
            print("np_train_path: {}".format(np_train_path))
            print("np_label_train_path :{}".format(np_label_train_path))
# ------------ANALYSIS PART-----------------------------------------------------------------------------
            if analysis_mode == True: #calculates statistics
                # calculates the characteristic parameters of blocks of sequences (same IC and same load cond)
                individual_train_scaler_params = {}
                individual_label_scaler_params = {}

                individual_train_scaler = StandardScaler()
                individual_label_scaler = StandardScaler()

                individual_train_scaler.fit(train_df_as_np_array)
                individual_label_scaler.fit(label_train_df_as_np_array)
                # print(individual_train_scaler.mean_, individual_train_scaler.scale_, individual_train_scaler.var_, individual_train_scaler.std_)

                individual_train_scaler_params['mean'] = np.ndarray.tolist(individual_train_scaler.mean_)
                individual_train_scaler_params['scale'] = np.ndarray.tolist(individual_train_scaler.scale_)
                # individual_train_scaler_params['std'] = np.ndarray.tolist(individual_train_scaler.std_)
                individual_train_scaler_params['var'] = np.ndarray.tolist(individual_train_scaler.var_)

                individual_label_scaler_params['mean'] = np.ndarray.tolist(individual_label_scaler.mean_)
                individual_label_scaler_params['scale'] = np.ndarray.tolist(individual_label_scaler.scale_)
                # deprecated individual_label_scaler_params['std'] = np.ndarray.tolist(individual_label_scaler.std_)
                individual_label_scaler_params['var'] = np.ndarray.tolist(individual_label_scaler.var_)

                # nested dict.
                seq_individual_params["sequence_" + identifier + "_" + str(j) + "_" + str(i) + ".npy"] = individual_train_scaler_params
                seq_individual_params["sequence_" + identifier + "_" + str(j) + "_" + str(i) + "_label_.npy"] = individual_label_scaler_params
# ------------END OF ANALYSIS PART----------------------------------------------------------------------
            if save_arrays == False:
                np.save(np_train_path, train_df_as_np_array)
                np.save(np_label_train_path, label_train_df_as_np_array)
            j = j + 1

        print(seq_length_dict) #these are of individual sequence lengths.
        json.dump(seq_length_dict, open(seq_length_dict_filename, 'wb'))
        json.dump(seq_group_params, open(seq_group_params_filename, 'wb'))
        json.dump(seq_individual_params, open(seq_individual_params_filename, 'wb'))

if __name__ == "__main__":
    parse_scattergro(save_arrays=False,analysis_mode=True)

