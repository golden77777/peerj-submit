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
#%matplotlib inline
from PIL import Image
from pandas import DataFrame as df
from pandas import DataFrame
import math
import torch
import torch.cuda
from torch.autograd import Variable
import pytorch_colors as colors
from torchvision import transforms
from torchvision.utils import save_image

q = 10000

l = 0
o = int(q / 50) - 100

for l in range(o):
    list = []
    n = l * 50
    for m in range(n, n+1250, 250):
        pn = np.array(Image.open("images2/aa_final_phones_acceletor_colorimage_"+"{0:06d}".format(m)+".png"))
        print(pn.shape)
        #pn = pn.transpose(2, 1, 0)
        pn = np.reshape(pn, (1, 1, 1250, 3))
        #pn = cv2.imread("images2/aa_final_phones_acceletor_colorimage_"+"{0:06d}".format(m)+".png")
        pn = torch.from_numpy(pn.astype(np.float32)).clone()
        print(pn.shape)
        img_hsv = colors.rgb_to_xyz(pn)
        img1 = img_hsv[0]
        print(img1.shape)
        save_image(img1, "image_ynv/aa_final_phones_acceletor_colorimage_"+"{0:06d}".format(m)+".png")

        #img_hsv  = img_hsv.to('cpu').detach().numpy().copy()
        #qn = cv2.imread("images_xyz/aa_final_phones_acceletor_colorimage_"+"{0:06d}".format(m)+".png", img_hsv)
