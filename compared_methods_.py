#必要なライブラリのインポート
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import cv2
import time
import argparse
import chainer
from chainer import links as L
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
#%matplotlib inline
from PIL import Image
from pandas import DataFrame as df
from pandas import DataFrame
import math
import json
from collections import OrderedDict
import pprint
import requests
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
import os
import re
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
#%matplotlib inline
import chainer.links as L
import random
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable
import chainer.functions as F
# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import model_selection, preprocessing, decomposition #機械学習用のライブラリを利用

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

df7 = pd.read_csv('converted_modify_200_true_new.csv')
df7

X = df7.iloc[:, 0:3750]
y = df7.iloc[:, 3750:3751]

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 解説4：主成分分析を実施-------------------------------
pca = decomposition.PCA(n_components=100)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.model_selection import StratifiedKFold

# pipe_lr = make_pipeline(StandardScaler(),SVC(kernel='linear'))
#
# kfold = StratifiedKFold(n_splits=5, random_state=1).split(X_train, y_train)
# scores = []
#
# for k, (train, test) in enumerate(kfold):
#     pipe_lr.fit(X_train[train], y_train[train])
#     score = pipe_lr.score(X_train[test], y_train[test])
#     scores.append(score)
#     print('Fold: %2d, class dist.: %s, Acc: %.3f' %
#         (k+1,np.bincount(y_train[train]), score))

from sklearn.svm import SVC
model = SVC(kernel='linear')
#「kernel」を変えることで、非線形SVMも実装可能

model.fit(X_train, y_train.values.ravel())

y_p = model.predict(X_test)


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_p, average="macro")
print("線形SVM")
print('精度，再現率，F値:',precision_recall_fscore_support(y_test, y_p, average="macro"))

# ランダムフォレストとの比較
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(min_samples_leaf=3)

model.fit(X_train, y_train.values.ravel())

y_p = model.predict(X_test)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_p, average="macro")
print("ランダムフォレスト")
print('精度，再現率，F値:',precision_recall_fscore_support(y_test, y_p, average="macro"))


# k近傍法との比較
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train.values.ravel())

y_p = model.predict(X_test)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_p, average="macro")
print("KNN")
print('精度，再現率，F値:',precision_recall_fscore_support(y_test, y_p, average="macro"))

#非線形SVMとの比較
from sklearn.svm import SVC
model = SVC(kernel='rbf')
#「kernel」を変えることで、非線形SVMも実装可能

model.fit(X_train, y_train.values.ravel())

y_p = model.predict(X_test)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_p, average="macro")
print("非線形SVM")
print('精度，再現率，F値:',precision_recall_fscore_support(y_test, y_p, average="macro"))


# ランダムフォレストとの比較
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(min_samples_leaf=3)

model.fit(X_train, y_train)

y_p = model.predict(X_test)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_p, average="macro")
print('精度，再現率，F値:',precision_recall_fscore_support(y_test, y_p, average="macro"))
