#必要なライブラリのインポート
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
%matplotlib inline
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
from sklearn.cross_validation import train_test_split
import os
import re
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
%matplotlib inline
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
from __future__ import print_function
available = True
import chainer.functions as F
import cupy



def load_image(img_path, augmentation=False, size=(224, 224)): #パス通す，サイズを決定する．
    img = Image.open(img_path)
    # 短辺長を基準とした正方形の座標を得る
    x_center = img.size[0] // 2
    y_center = img.size[1] // 2
    half_short_side = max(x_center, y_center)
    x0 = x_center - half_short_side
    y0 = y_center - half_short_side
    x1 = x_center + half_short_side
    y1 = y_center + half_short_side
    img = img.crop((x0, y0, x1, y1))
    img = img.resize(size)
    img = np.array(img, dtype=np.float32)
    return img


#ある一つの画像の読み込みを行う
img_true = load_image('100ms_XYZ_bike/aaaa_final_phones_acceletor_colorimage_1681000.png', augmentation=False)
plt.imshow(img_true / 255.)


#ある一つの画像の読み込みを行う
img_true = load_image('100ms_XYZ_walk/aaaa_final_phones_acceletor_colorimage_1601000.png', augmentation=False)
plt.imshow(img_true / 255.)

#ある一つの画像の読み込みを行う
img_true = load_image('100ms_XYZ_stand/aaaa_final_phones_acceletor_colorimage_1600200.png', augmentation=False)
plt.imshow(img_true / 255.)


#ある一つの画像の読み込みを行う
img_true = load_image('100ms_XYZ_bike/aaaa_final_phones_acceletor_colorimage_1500200.png', augmentation=False)
plt.imshow(img_true / 255.)


#画像のサイズを統一するための操作
import numpy as np

def load_image(img_path, augmentation=False, size=(224, 224)): #パス通す，サイズを決定する．
    img = Image.open(img_path)
    # 短辺長を基準とした正方形の座標を得る
    x_center = img.size[0] // 2
    y_center = img.size[1] // 2
    half_short_side = max(x_center, y_center)
    x0 = x_center - half_short_side
    y0 = y_center - half_short_side
    x1 = x_center + half_short_side
    y1 = y_center + half_short_side
    img = img.crop((x0, y0, x1, y1))
    img = img.resize(size)
    img = np.array(img, dtype=np.float32)
    return img


#　実際にどのような画像に変換されているか確認する
plt.figure(figsize=(20, 20))
for i in range(20):
    imge_id = i*100
    imge_file = '100ms_XYZ_stand/aaaa_final_phones_acceletor_colorimage_{0:07d}.png'.format(imge_id)
    imge = Image.open(imge_file)
    plt.subplot(4, 5, i+1)
    plt.title(imge_file)
    plt.imshow(imge)


#ある一つの画像の読み込みを行う
img_true = load_image('100ms_XYZ_stand/aaaa_final_phones_acceletor_colorimage_0008000.png', augmentation=False)
plt.imshow(img_true / 255.)

#CNNモデルに正確に学習させるために，channel軸を一番初めに持ってくる
print(img_true.shape) # (0:height, 1:width, 2:channel)
img_true = img_true.transpose(2, 0, 1) # <= 軸を2, 0, 1の順番に並び替える。
print(img_true.shape) #変換後の結果

#batch次元を追加し，4次元配列に変換する
input_data = img_true.reshape(1, 3, 224, 224)

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


X = []
Y = []

# 対象Aの画像
for picture in list_pictures('100ms_RGB_bike_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(0)

# 対象Aの画像
for picture in list_pictures('100ms_RGB_sit_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(1)

# 対象Aの画像
for picture in list_pictures('100ms_RGB_stairsdown_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(2)

# 対象Bの画像
for picture in list_pictures('100ms_RGB_stairsup_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(3)


# 対象Bの画像
for picture in list_pictures('100ms_RGB_stand_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(4)


# 対象Bの画像
for picture in list_pictures('100ms_RGB_walk_2/'):
    img = img_to_array(load_img(picture, target_size=(224,224)))
    X.append(img)
    Y.append(5)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=111)


a1 = X[0:7000]
a2 = X[7000:14000]
a3 = X[14000:21000]
a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)

a = [a1, a2, a3]

b1 = Y[0:7000]
b2 = Y[7000:14000]
b3 = Y[14000:21000]
b1 = np.array(b1)
b2 = np.array(b2)
b3 = np.array(b3)

b = [b1, b2, b3]

c1 = X_test[0:7000]
c2 = X_test[7000:14000]
c3 = X_test[14000:21000]
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)

c = [c1, c2, c3]

d1 = y_test[0:7000]
d2 = y_test[7000:14000]
d3 = y_test[14:21000]
d1 = np.array(d1)
d2 = np.array(d2)
d3 = np.array(d3)

d = [d1, d2, d3]


class LeNet(chainer.Chain):
    def __init__(self, train=True):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 6, 5, stride=1)
            self.conv2=L.Convolution2D(None, 16, 5, stride=1)
            self.fc3=L.Linear(None, 120)
            self.fc4=L.Linear(None, 64)
            self.fc5=L.Linear(None, 6)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv2(h))), 2, stride=2)
        h = F.sigmoid(self.fc3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)

        return h


# CNN学習のエポック数の設定
n_epoch = 1
batch_size = 70

#モデルを関数の形に再構築
model = LeNet()

# GPUで学習させる場合はモデルをGPUに送る
model.to_gpu(0)

model.xp

#学習パラメータの調整
opt = optimizers.MomentumSGD(lr=1e-2)
opt.setup(model)
model.fc5.W.update_rule.hyperparam.lr = 1e-2
model.fc5.b.update_rule.hyperparam.lr = 1e-2

#各エポックにおけるコスト関数の値
train_loss = []
train_acc = []
test_loss = []
test_acc = []

#ミニバッチのシャッフルを行う
perm = np.random.permutation(100)
perm[0:16]

#トレーニングデータの一覧
#X_train[perm[0:16]]


#実際の学習を始める
for epoch in range(n_epoch):
    print('epoch: {}'.format(epoch))

    # 学習
    print('train')
    perm = np.random.permutation(len(X_train))
    sum_loss = 0.
    sum_acc = 0.
    for i in range(0, len(perm), batch_size):
        x_batch = []
        t_batch = []
        for j in range(batch_size):
            r = random.randrange(2)
            p = random.randrange(7000)
            img =  a[r][p]
            img = img.transpose(2, 0, 1)
            img = img[::-1]
            x_batch.append(img)
            t = b[r][p]
            t_batch.append(t)
        # prepare minibatch
        #x_batch, t_batch = make_batch(train_data[perm[i:i+batch_size]], augmentation=True)
        x_batch = model.xp.array(x_batch) # model.xpはmodelがGPUにあるときはcupy, CPUにあるときはnumpyになる。
        t_batch = model.xp.array(t_batch)

        # forward
        y = model(x_batch)#, train=True)
        loss = F.softmax_cross_entropy(y, t_batch)
        acc = F.accuracy(y, t_batch)

        # backward and update
        model.cleargrads()
        loss.backward()
        opt.update()

        # lossとaccuracyの記録
        # これらはバッチサイズで割られた値なので評価時と正しく比較するためにミニバッチのデータ数をかけて補正しておく
        sum_loss += cuda.to_cpu(loss.data) * len(x_batch)
        sum_acc += cuda.to_cpu(acc.data) * len(x_batch)

    train_loss.append(sum_loss / len(X_train)) # 全データ数で割る
    train_acc.append(sum_acc / len(X_train))

    # このepochのloss, accuracyの和を表示
    print(train_loss[-1], train_acc[-1])

    # 評価
    print('test')
    sum_loss = 0.
    sum_acc = 0.
    for i in range(0, len(X_test), batch_size):
        x_batch = []
        t_batch = []
        for j in range(batch_size):
            q = random.randrange(2)
            r = random.randrange(7000)
            img =  c[q][r]
            img = img.transpose(2, 0, 1)
            img = img[::-1]
            x_batch.append(img)
            t = d[q][r]
            t_batch.append(t)

#         x_batch =  c[q][i:i+batch_size]
#         x_batch =  x_batch.reshape(batch_size, 3, 224, 224)
#         t_batch =  d[q][i:i+batch_size]
        #x_batch, t_batch = make_batch(test_data[i:i+batch_size], augmentation=False)
        x_batch = model.xp.array(x_batch)
        t_batch = model.xp.array(t_batch)
        b = x_batch.shape[0]

        # forward
        y = model(x_batch)#, train=True)
        loss = F.softmax_cross_entropy(y, t_batch)
        acc = F.accuracy(y, t_batch)

        # lossとaccuracyの記録
        sum_loss += cuda.to_cpu(loss.data) * len(x_batch)
        sum_acc += cuda.to_cpu(acc.data) * len(x_batch)

    test_loss.append(sum_loss / len(X_test))
    test_acc.append(sum_acc / len(X_test))

    # このepochのloss, accuracyの和を表示
    print(test_loss[-1], test_acc[-1])
