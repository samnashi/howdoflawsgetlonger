from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, FastICA, FactorAnalysis, IncrementalPCA
from corpus_characterizer import generator_chunker
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
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
pca_dict = {}

for index_to_load in range(0,len(combined_filenames)):
#for index_to_load in range(0,2):
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

    #train_array=train_array[-10000:,:] #cut it short just to test

    #KernelPCA, FastICA, FactorAnalysis, IncrementalPCA

    #tsne = TSNE(n_components=2)
    # train_embedded = tsne.fit_transform(train_array)

    # tsvd = TruncatedSVD(n_components=2)
    # train_embedded = tsvd.fit_transform(train_array)
    plt.clf()
    plt.cla()

    pca = PCA(n_components=3,svd_solver='full')
    train_embedded_pca = pca.fit_transform(train_array)
    print('pca')
    print(pca.components_.shape,pca.noise_variance_.shape,pca.explained_variance_.shape,
          pca.explained_variance_ratio_.shape)
    print(pca.components_)
    print(pca.noise_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(train_embedded_pca.shape, type(train_embedded_pca))
    plt.subplot(231)
    plt.plot(train_embedded_pca)
    colnames = ['component_of_' + str(i) for i in range(0,11)]
    colnames.append(['explained_variance','explained_variance_ratio'])
    pca_combined_results_shape = (pca.components_.shape[0],(pca.components_.shape[1] + pca.explained_variance_.ndim
                                                            + pca.explained_variance_ratio_.ndim))

    pca_results_ndarray = np.ndarray(shape=pca_combined_results_shape)
    pca_results_ndarray[:, 0: pca.components_.shape[1]] = pca.components_
    pca_results_ndarray[:, pca.components_.shape[1]:pca.components_.shape[1] + pca.explained_variance_.ndim] = \
        np.reshape(np.asarray(pca.explained_variance_),newshape=(pca.explained_variance_.shape[0],1))
    pca_results_ndarray[:, pca.components_.shape[1] + pca.explained_variance_.ndim:] = \
        np.reshape(np.asarray(pca.explained_variance_ratio_),newshape=(pca.explained_variance_ratio_.shape[0],1))
    print(pca_results_ndarray.shape)
    pca_dict[files[0]] = pca_results_ndarray.flatten()

#     kpca = KernelPCA(n_components=3)
#     train_embedded_kpca = kpca.fit_transform(train_array)
#     print("kpca")
#     print(kpca.alphas_)
#     print(train_embedded_kpca.shape, type(train_embedded_kpca))
#     plt.subplot(232)
#     plt.plot(train_embedded_kpca)
#
#     fica = FastICA(n_components=3)
#     train_embedded_fica = fica.fit_transform(train_array)
#     print("fast ICA")
#     print(fica.components_,fica.mixing_)
#     print(train_embedded_fica.shape, type(train_embedded_fica))
#     print()
#     plt.subplot(233)
#     plt.plot(train_embedded_fica)
#
#     #train_embedded_fica = fica.fit_transform(train_array)
#     #components, mixing, n_iter
#
#     fa = FactorAnalysis(n_components=3)
#     train_embedded_fa = fa.fit_transform(train_array)
#     print("factor analysis")
#     print("noise variance: {}".format(fa.noise_variance_))
#     print(train_embedded_fa.shape, type(train_embedded_fa))
#     plt.subplot(234)
#     plt.plot(train_embedded_fa)
#
#     ipca = IncrementalPCA(n_components=3)
#     print("incremental pca")
#     train_embedded_ipca = ipca.fit_transform(train_array)
#     print(train_embedded_ipca.shape, type(train_embedded_ipca))
#     plt.subplot(235)
#     plt.plot(train_embedded_ipca)
#
#     #print(train_embedded_ipca.shape)
#     plt.show()
#     plt.savefig("./dimred_figs.jpg")
#     '''Attributes:
# components_ : array, [n_components, n_features], Components with maximum variance.
# loglike_ : list, [n_iterations], The log likelihood at each iteration.
# noise_variance_ : array, shape=(n_features,), The estimated noise variance for each feature.
# n_iter_ : int, Number of iterations run.
# '''
print(pca_dict)
pca_df = pd.DataFrame.from_dict(pca_dict,orient='index')
pca_df.to_csv('./analysis/pca_df.csv')
(pca_df.describe()).to_csv('./analysis/pca_df_describe.csv')
    # plot_embedding(train_embedded,
    #                "t-SNE embedding of the digits (time %.2fs)" %
    #                (time() - t0))