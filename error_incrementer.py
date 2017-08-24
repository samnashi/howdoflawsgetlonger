import numpy as np
import json


#----------------------------------------------------------------------------------------------------------------
fname_entire = ""
fname_group = ""
fname_individual = ""
calculate_all = True
#---------------------------------------------------------------------------------------------------------------------

standardscaler_coeffs_dict = {}
seq_entire_params_filename = fname_entire
if fname_entire == "":
    seq_entire_params_filename = "./analysis/seq_entire_params.json"
seq_entire_params = json.load(open(seq_entire_params_filename))

# usage print(seq_entire_params[seq_entire_params.iterkeys().next()].keys())

seq_group_params_filename = fname_group
if fname_group == "":
    seq_group_params_filename = "./analysis/seq_group_params.json"
seq_group_params = json.load(open(seq_group_params_filename))

seq_individual_params_filename = fname_individual
if fname_individual == "":
    seq_individual_params_filename = "./analysis/seq_individual_params.json"
seq_individual_params = json.load(open(seq_individual_params_filename))

if calculate_all == True:
    '''right now, all this does is apply one aggregation method function on each dict, but this can be tweaked later '''
    # entire
    scale_entire_data_list = []  # this is the final values. cast it to a numpy array before returning.
    scale_entire_label_list = []
    # first item's shape. should be 11 cols.
    scale_entire_data_np = np.empty(shape=(np.asarray(seq_entire_params['data']['scale']).shape))
    scale_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['scale']).shape))

    mean_entire_data_list = []
    mean_entire_label_list = []
    mean_entire_data_np = np.empty(shape=(np.asarray(seq_entire_params['data']['mean']).shape))
    mean_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['mean']).shape))

    var_entire_data_list = []
    var_entire_label_list = []
    var_entire_data_np = np.empty(shape=(np.asarray(seq_entire_params['data']['var']).shape))
    var_entire_label_np = np.empty(shape=(np.asarray(seq_entire_params['label']['mean']).shape))

    for key_entire, values in seq_entire_params.iteritems():  # data or label
        print("key_entire: {}, values: {}".format(key_entire,values))
        for key_tier2 in values.iteritems():
            print("key_tier2: {}".format(key_tier2))
            if key_entire == 'data':
                if key_tier2 == 'scale':
                    scale_entire_data_list.append(seq_entire_params[key_tier2])
                if key_tier2 == 'mean':
                    mean_entire_data_list.append(seq_entire_params[key_tier2])
                if key_tier2 == 'var':
                    var_entire_data_list.append(seq_entire_params[key_tier2])
            if key_entire == 'label':
                if key_tier2 == 'scale':
                    scale_entire_label_list.append(seq_entire_params[key_tier2])
                if key_tier2 == 'mean':
                    mean_entire_label_list.append(seq_entire_params[key_tier2])
                if key_tier2 == 'var':
                    var_entire_label_list.append(seq_entire_params[key_tier2])

    for i in range(0,len(var_entire_data_list)):
        scale_entire_data_np[i,:] = scale_entire_data_list[i]
        mean_entire_data_np[i,:] = mean_entire_data_list[i]
        var_entire_data_np[i,:] = var_entire_data_list[i]
        scale_entire_label_np[i,:] = scale_entire_label_list[i]
        mean_entire_label_np[i,:] = mean_entire_label_list[i]
        var_entire_label_np[i,:] = var_entire_label_list[i]


    # print("the dict-read array shape's list: {}".format(np.asarray(seq_entire_params['data']['var']).shape))
    # print(np.asarray(seq_entire_params['data']['var']))
    # print(var_entire_data_np)
    # print("the list-derived array shape: {}".format(var_entire_data_np.shape))
    #assert var_entire_data_np.shape == np.asarray(seq_entire_params['data']['var']).shape

    # mean_entire_data
    initial_key = seq_group_params.keys()[2]
    assert ("_colnames") not in initial_key 
    #print("initial key: {}".format(initial_key))
    scale_group_data_list = []
    scale_group_data_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['train']['scale']).shape))
    mean_group_data_list = []
    mean_group_data_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['train']['mean']).shape))
    var_group_data_list = []
    var_group_data_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['train']['var']).shape))

    scale_group_label_list = []
    scale_group_label_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['label']['scale']).shape))
    mean_group_label_list = []
    mean_group_label_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['label']['mean']).shape))
    var_group_label_list = []
    var_group_label_np = np.empty(shape=(np.asarray(seq_group_params[initial_key]['label']['var']).shape))


    for key_csv, data_or_label in seq_group_params.iteritems():  # these are csv names.
        if key_csv != "label_colnames" and key_csv != "train_colnames":
            print("key_csv: {} \n, data_or_label: {}".format(key_csv, data_or_label))
            if data_or_label.keys()[0] == "train": #TODO data or label is a dict!!
                print("sgp train scale: {}".format(data_or_label['train']['scale']))
                scale_group_data_list.append(data_or_label['train']['scale'])

    for i in range(0, len(var_group_data_list)):
        scale_group_data_np[i, :] = scale_group_data_list[i]
        mean_group_data_np[i, :] = mean_group_data_list[i]
        var_group_data_np[i, :] = var_group_data_list[i]
        scale_group_label_np[i, :] = scale_group_label_list[i]
        mean_group_label_np[i, :] = mean_group_label_list[i]
        var_group_label_np[i, :] = var_group_label_list[i]
        
    print("dict-read array shape's list: {}".format(np.asarray(seq_group_params[initial_key]['train']['var']).shape))
    print(np.asarray(seq_group_params[initial_key]['train']['var']))
    print(var_group_data_np)
    print("the list-derived array shape: {}".format(var_group_data_np.shape))
            # print(data_or_label)
            # print(data_or_label.iteritems())
        # scale_group_data_np = np.empty(shape=(np.asarray(seq_group_params['train'][''])))
        #
        # # still need to go through the second tier
        # # TODO: group dict is also useful for finding out which group does the sequence belong to, for postmortem
        # pass

    # scale_individual_list = []
    # mean_individual_list = []
    # var_individual_list = []
    # for keys_group in seq_group_params.keys():
    #     # still need to go through
    #     pass
    #     # TODO numpy savetxt covariance matrices