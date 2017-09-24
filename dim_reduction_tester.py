from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, FastICA, FactorAnalysis, IncrementalPCA
from corpus_characterizer import generator_chunker
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from time import time

# def plot_embedding(X, title=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
#                  color=plt.cm.Set1(y[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#
#     if hasattr(offsetbox, 'AnnotationBbox'):
#         # only print thumbnails with matplotlib > 1.0
#         shown_images = np.array([[1., 1.]])  # just something big
#         for i in range(digits.data.shape[0]):
#             dist = np.sum((X[i] - shown_images) ** 2, 1)
#             if np.min(dist) < 4e-3:
#                 # don't show points that are too close
#                 continue
#             shown_images = np.r_[shown_images, [X[i]]]
#             imagebox = offsetbox.AnnotationBbox(
#                 offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                 X[i])
#             ax.add_artist(imagebox)
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)


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
for index_to_load in range(0,1):
    files = combined_filenames[index_to_load]
    print("files: {}".format(files))
    data_load_path = train_path + '/data/' + files[0]
    label_load_path = train_path + '/label/' + files[1]
    train_array = np.load(data_load_path)
    label_train_array = np.load(label_load_path)
    if train_array.shape[1] > 11:
        train_array = train_array[:,1:]
    if label_train_array.shape[1] > 5:
        label_train_array = label_train_array[:,1:]
    identifier = files[0][:-4]

    train_array=train_array[-10000:,:] #cut it short just to test

    #KernelPCA, FastICA, FactorAnalysis, IncrementalPCA
    # train_embedded = TSNE(n_components=2).fit_transform(train_array)
    #train_embedded = TruncatedSVD(n_components=2).fit_transform(train_array)
    #train_embedded = PCA(n_components=3).fit_transform(train_array)
    train_embedded = KernelPCA(n_components=4).fit_transform(train_array)
    #train_embedded = FastICA(n_components=2).fit_transform(train_array)
    #train_embedded = FactorAnalysis(n_components=2).fit_transform(train_array)
    #train_embedded = IncrementalPCA(n_components=2).fit_transform(train_array)
    print(train_embedded.shape)
    plt.clf()
    plt.cla()
    plt.plot(train_embedded)
    plt.show()

    # plot_embedding(train_embedded,
    #                "t-SNE embedding of the digits (time %.2fs)" %
    #                (time() - t0))