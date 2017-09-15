import numpy as np
import os


conversion_folder = "/home/efi/Documents/conversions/"

#@@@@@@@@@@@@@@ RELATIVE PATHS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Base_Path = "./"
image_path = "./images/"
train_path = "./train/"
test_path = "./test/"
analysis_path = "./analysis/"
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
desired_colnumber = 14
#TODO: try with 15.
header_corrcoef = "first 9: all but percent_damage and step_index. last four: deltaA 1-2-3-4"

length_total = 0
for index_to_load in range(0,len(combined_filenames)):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    identifier = files[0][:-4]
    length_total += train_array.shape[0]
    #BLOCK FOR AGGREGATED SEQUENCES
print("total length = {}".format(length_total))

for index_to_load in range(0,len(combined_filenames)):
    files = combined_filenames[index_to_load]
    identifier = files[0][:-4]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    if train_array.shape[1] > 11:
        train_array = train_array[:,1:]
    if label_train_array.shape[1] > 5:
        label_train_array = label_train_array[:,1:]
    train_array_filename = conversion_folder + identifier + ".csv"
    label_train_array_filename = conversion_folder + identifier + "label.csv"
    np.savetxt(fname=train_array_filename,X=train_array,delimiter=",")
    np.savetxt(fname=label_train_array_filename,X=label_train_array,delimiter=",")
    print(("data and label as csv for {} saved.").format(identifier))
