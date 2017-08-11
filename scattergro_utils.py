import json

def get_shortest_length(isTestingPhase=False):

    test_path = "/home/ihsan/Documents/thesis_models/test/"
    train_path = "/home/ihsan/Documents/thesis_models/train/"
    if isTestingPhase == False:
        seq_length_dict_file = train_path + "sequence_lengths.json"
    if isTestingPhase == True:
        seq_length_dict_file = test_path + "sequence_lengths.json"
    #print(seq_length_dict_file)

    seq_length_dict = json.load(open(seq_length_dict_file))
    #print seq_length_dict

    lengths = list(seq_length_dict.values())
    lengths.sort()
    print(lengths)
    biggest_possible_batch_size = min(lengths)
    print("nearest int: {}".format(([int(i / biggest_possible_batch_size) for i in lengths])))
    print("modulos: {}".format([i % biggest_possible_batch_size for i in lengths]))
    return biggest_possible_batch_size

#get_shortest_length()


