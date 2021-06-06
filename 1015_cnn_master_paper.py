#必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

#画像の次元を変換する上で必要なOpenCVのインポート
import cv2

# chainer系ライブラリをインポートする
import time
import argparse
import numpy as np
import chainer
from chainer import links as L
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

#グラフ描画，画像変換に必要なライブラリのインポート
%matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt

#必要なライブラリのインポート
import cv2
import pandas as pd
import numpy as np
from pandas import DataFrame as df
from pandas import DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import math
import cupy
df = pd.read_csv('aa_final_phones_acceletor.csv')
i = 0
for i in range(0, 936000, 50):
    #特定の列のみを抽出する

    df.iloc[i:i+250, [0, 1, 2]]
    dfi = df.iloc[i:i+250, [0, 1, 2]]
    #データの種類ごとに変数をまとめる
    x1 = dfi.iloc[:, 0]
    x2 = dfi.iloc[:, 1]
    x3 = dfi.iloc[:, 2]

    # 各種パラメータを正規化する
    def min_max_normalization(x1):
        x1_min = x1.min()
        x1_max = x1.max()
        x1_norm = (x1 - x1_min) / ( x1_max - x1_min)
        #x1_norm = 0.75*x1_norm
        return x1_norm

    def min_max_normalization(x2):
        x2_min = x2.min()
        x2_max = x2.max()
        x2_norm = (x2 - x2_min) / ( x2_max - x2_min)
        return x2_norm

    def min_max_normalization(x3):
        x3_min = x3.min()
        x3_max = x3.max()
        x3_norm = (x3 - x3_min) / ( x3_max - x3_min)
        return x3_norm

    #色変換するために、RGB255倍する
    y1 = 255*min_max_normalization(x1)
    y2 = 255*min_max_normalization(x2)
    y3 = 255*min_max_normalization(x3)

    #変更した数値を一つの配列にまとめる
    df_i= np.array([y1, y2, y3])

    #そのままでは分析ができないので、縦型に変換する
    df_i = df_i.T

    import csv
    csv_obj = df_i
    data = [ v for v in csv_obj]
    data_conved = [[float(elm) for elm in v] for v in data]
    print(data_conved)

    for j in range(250):
        pj = data_conved[j]
        qj = map(int, pj)
        imgj = Image.new('RGB', (1, 1), 'white')
        pixj = imgj.load()
        (r, g, b) =  qj
        pixj[0, 0] = (r, g, b)
        imgj.save('medium.png')
        imgj = Image.open('medium.png')
        imgj = imgj.resize((50, 1))
        imgj.save("colorfigure"+"{0:04d}".format(j)+".png")

    k = 0
    list = []
    for k in range(250):
        tk = cv2.imread("colorfigure"+"{0:04d}".format(k)+".png")
        list.append(tk)

    im_h = cv2.vconcat(list)
    cv2.imwrite("colorimage_"+"{0:06d}".format(i)+".png", im_h)

l = 0
for l in range(9340):
    list = []
    n = l*100
    for m in range(n, n+1250, 250):
        pn = cv2.imread("colorimage_"+"{0:06d}".format(m)+".png")
        pn = cv2.cvtColor(pn, cv2.COLOR_RGB2XYZ)
        list.append(pn)
        im_v = cv2.hconcat(list)
        cv2.imwrite("aa_final_phones_acceletor_colorimage_"+"{0:06d}".format(n)+".png", im_v)

!rm colorimage_*.png
!rm colorfigure*.png



#必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

#画像の次元を変換する上で必要なOpenCVのインポート
import cv2

# chainer系ライブラリをインポートする
import time
import argparse
import numpy as np
import chainer
from chainer import links as L
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

#グラフ描画，画像変換に必要なライブラリのインポート
%matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt

#必要なライブラリのインポート
import cv2
import pandas as pd
import numpy as np
from pandas import DataFrame as df
from pandas import DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import math
import cupy


# 念のため，画像の大きさで集計をかける
w_list, h_list, rate_list = [], [], []
for i in range(9340):
    img_id = i*100
    img_file = '1009_RGB_0.1s_/aa_final_phones_acceletor_colorimage_'+'{0:06d}.png'.format(img_id)
    img = Image.open(img_file)
    w, h = img.size
    w_list.append(w)
    h_list.append(h)
    rate_list.append(w / h)


#CNNモデルに学習させられるように画像サイズを変換

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
    imge_file = '1009_RGB_0.1s_/aa_final_phones_acceletor_colorimage_{0:06d}.png'.format(imge_id)
    imge = Image.open(imge_file)
    plt.subplot(4, 5, i+1)
    plt.title(imge_file)
    plt.imshow(imge)

#CNNモデルに学習させられるように画像サイズを変換

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


#ある一つの画像の読み込みを行う
img_true = load_image('1009_RGB_0.1s_/aa_final_phones_acceletor_colorimage_000800.png', augmentation=False)
plt.imshow(img_true / 255.)

#CNNモデルに正確に学習させるために，channel軸を一番初めに持ってくる
print(img_true.shape) # (0:height, 1:width, 2:channel)
img_true = img_true.transpose(2, 0, 1) # <= 軸を2, 0, 1の順番に並び替える。
print(img_true.shape) #変換後の結果

#batch次元を追加し，4次元配列に変換する
input_data = img_true.reshape(1, 3, 224, 224)

#CNNを構築するための環境構築を行う
import chainer
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
import collections
import os
import numpy
from PIL import Image
available = True


import chainer
import chainer.functions as F
import chainer.links as L

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

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import cuda
from chainer import optimizers
import cupy

lenet = LeNet()

lenet(input_data)

prob = lenet(input_data)

prob.shape

lenet._children

model = LeNet()

def make_batch(img_path_list, augmentation=True):
    x_batch = []
    t_batch = []
    for img_path in img_path_list:
        img = load_image(img_path, augmentation=augmentation)
        img = img.transpose(2, 0, 1)
        img = img[::-1]
#        img -= np.array([103.939, 116.779, 123.68],
#                        dtype=np.float32).reshape(3, 1, 1)
        x_batch.append(img)
        i = int(img_path[-10:-4])
        if i in range(0,55162):
            t_batch.append(0)
        elif i in range(55163,115116):
            t_batch.append(1)
        elif i in range(115116,166059):
            t_batch.append(2)
        elif i in range(166060,167898):
            t_batch.append(3)
        elif i in range(167899,178673):
            t_batch.append(4)
        elif i in range(178674,188155):
            t_batch.append(3)
        elif i in range(188156,196812):
            t_batch.append(4)
        elif i in range(196813,207216):
            t_batch.append(3)
        elif i in range(207217,216091):
            t_batch.append(4)
        elif i in range(2161092,226846):
            t_batch.append(3)
        elif i in range(226847,236325):
            t_batch.append(4)
        elif i in range(236326,247019):
            t_batch.append(3)
        elif i in range(247020,254223):
            t_batch.append(4)
        elif i in range(254224,290899):
            t_batch.append(5)
        elif i in range(290900,347665):
            t_batch.append(0)
        elif i in range(347666,407260):
            t_batch.append(1)
        elif i in range(407261,455982):
            t_batch.append(2)
        elif i in range(455983,458684):
            t_batch.append(3)
        elif i in range(458685,467461):
            t_batch.append(4)
        elif i in range(467462,477154):
            t_batch.append(3)
        elif i in range(477155,484093):
            t_batch.append(4)
        elif i in range(484094,494741):
            t_batch.append(3)
        elif i in range(494742,503219):
            t_batch.append(4)
        elif i in range(503220,513974):
            t_batch.append(3)
        elif i in range(513975,521459):
            t_batch.append(4)
        elif i in range(521460,531048):
            t_batch.append(3)
        elif i in range(531049,537939):
            t_batch.append(4)
        elif i in range(537940,581439):
            t_batch.append(5)
        elif i in range(581440,609090):
            t_batch.append(0)
        elif i in range(609091,639211):
            t_batch.append(1)
        elif i in range(639212,663579):
            t_batch.append(2)
        elif i in range(663580,664460):
            t_batch.append(3)
        elif i in range(664461,669660):
            t_batch.append(4)
        elif i in range(669660,674450):
            t_batch.append(3)
        elif i in range(674460,678830):
            t_batch.append(4)
        elif i in range(678830,684180):
            t_batch.append(3)
        elif i in range(684180,688440):
            t_batch.append(4)
        elif i in range(688450,693730):
            t_batch.append(3)
        elif i in range(693740,697770):
            t_batch.append(4)
        elif i in range(697780,703050):
            t_batch.append(3)
        elif i in range(703060,706630):
            t_batch.append(4)
        elif i in range(706640,729460):
            t_batch.append(5)
        elif i in range(729470,757600):
            t_batch.append(0)
        elif i in range(757610,787480):
            t_batch.append(1)
        elif i in range(787490,811300):
            t_batch.append(2)
        elif i in range(811310,812640):
            t_batch.append(3)
        elif i in range(812650,817420):
            t_batch.append(4)
        elif i in range(817430,822220):
            t_batch.append(3)
        elif i in range(822230,825760):
            t_batch.append(4)
        elif i in range(825770,831120):
            t_batch.append(3)
        elif i in range(831130,834570):
            t_batch.append(4)
        elif i in range(834580,839900):
            t_batch.append(3)
        elif i in range(839910,843850):
            t_batch.append(4)
        elif i in range(843860,849150):
            t_batch.append(3)
        elif i in range(849160,852560):
            t_batch.append(4)
        elif i in range(852570,874100):
            t_batch.append(5)
        elif i in range(874110,900260):
            t_batch.append(0)
        elif i in range(900270,930240):
            t_batch.append(1)
        else:
            t_batch.append(2)
    return np.array(x_batch, dtype=np.float32), np.array(t_batch, dtype=np.int32)# 教師データをnumpy arrayに渡す必要がある

print(F.softmax(model(input_data)))

import glob
img_list = glob.glob('1009_RGB_0.1s_/aa_final_phones_acceletor*.png')


# 特定の二つの画像に関して，入力変数，出力変数を特定する
x_batch, t_batch = make_batch(['1009_RGB_0.1s_/aa_final_phones_acceletor_colorimage_800000.png', '1009_RGB_0.1s_/aa_final_phones_acceletor_colorimage_910000.png'])

#入力変数のサイズを確認する
x_batch.shape

#t_batchの中身を見る
t_batch

#適当に10個のデータを確認
img_list[:10]

#データの総数を確認
print(len(img_list))

#データをトレーニング，テストデータに分類
# from numpy.random import *
# from numpy import *
# city = [0,1,2,3,4,5,6,7,8,9]
# a = random.choice(city,10,replace=False) # 5個をランダム抽出（重複なし)

#a = [7,1]
#a = [9,3]
a = [0,2]
#a = [6,8]
#a = [4,5]

# 実際のデータの分割
train_data = []
test_data = []
for img_path in img_list:
    img_id = int(img_path[-10:-6])
    if img_id % 10 in a:
        test_data.append(img_path)
    else:
        train_data.append(img_path)

#トレーニングデータ，テストデータの数を確認
len(train_data), len(test_data)

#トレーニングデータとテストデータを配列に変換する
train_data = np.array(train_data)
test_data = np.array(test_data)

# トレーニングデータの中身をみる
train_data

#トレーニングデータより3, 5, 6番目を取り出す
train_data[[3, 5, 6]]

# CNN学習のエポック数の設定
n_epoch = 60
batch_size = 70

#モデルを関数の形に再構築
model = LeNet()

# GPUで学習させる場合はモデルをGPUに送る
model.to_gpu()

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

#トレーニングデータの一覧
train_data[perm[0:16]]

#実際の学習を始める
for epoch in range(n_epoch):
    print('epoch: {}'.format(epoch))

    # 学習
    print('train')
    perm = np.random.permutation(len(train_data))
    sum_loss = 0.
    sum_acc = 0.
    for i in range(0, len(perm), batch_size):
        # prepare minibatch
        x_batch, t_batch = make_batch(train_data[perm[i:i+batch_size]], augmentation=True)
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

    train_loss.append(sum_loss / len(train_data)) # 全データ数で割る
    train_acc.append(sum_acc / len(train_data))

    # このepochのloss, accuracyの和を表示
    print(train_loss[-1], train_acc[-1])

    # 評価
    print('test')
    sum_loss = 0.
    sum_acc = 0.
    for i in range(0, len(test_data), batch_size):
        x_batch, t_batch = make_batch(test_data[i:i+batch_size], augmentation=False)
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

    test_loss.append(sum_loss / len(test_data))
    test_acc.append(sum_acc / len(test_data))

    # このepochのloss, accuracyの和を表示
    print(test_loss[-1], test_acc[-1])

# Accuracy
plt.figure(figsize=(8, 5))
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(train_acc)
plt.plot(test_acc)

# Loss
plt.figure(figsize=(8, 5))
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_loss)
plt.plot(test_loss)

ys = []
ts = []
for i in range(0, len(test_data), batch_size):
    x_batch, t_batch = make_batch(test_data[i:i+batch_size], augmentation=False)
    x_batch = model.xp.array(x_batch)
    t_batch = model.xp.array(t_batch)
    y = model(x_batch)
    y = F.softmax(y)
    ys.append(cuda.to_cpu(y.data))
    ts.append(cuda.to_cpu(t_batch))

ys = np.concatenate(ys)
ts = np.concatenate(ts)

pred = np.argmax(ys, axis=1)

pred

ts

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(ts, pred, average="macro")
print('精度，再現率，F値:',precision_recall_fscore_support(ts, pred, average="macro"))


from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

confmat = metrics.confusion_matrix(ts, pred)


classes = ['stand', 'sit', 'walk', 'stairup',
           'stairdown', 'bike']

plt.figure(figsize=(12, 12))
plot_confusion_matrix(confmat, classes=classes, normalize=True)
