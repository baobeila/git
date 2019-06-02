#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: phpergao
@license: Apache Licence 
@contact: endoffight@gmail.com
@site: http://www.phpgao.com
@software: PyCharm
@file: 23.py
@time: 2019/5/19 22:27
"""

import scipy.io
import numpy as np
import os
import scipy.misc
path = r'D:\pycharm\Pycharmworkspace\GitHub\neural-style-master\neural-style-master\imagenet-vgg-verydeep-19.mat'
vgg = scipy.io.loadmat(path)
print(type(vgg))#dict
#dict_keys(['__header__', '__version__', '__globals__', 'layers', 'classes', 'normalization'])
print(vgg.keys())
print(vgg['__header__'])#MATLAB 5.0 MAT-file Platform: posix, Created on: Sat Sep 19 12:27:40 2015'
print(vgg['__version__'])#1.0
print(vgg['__globals__'])#[]
# print(vgg['classes'])#array(['n13054560'], dtype='<U9'),
# print(vgg['normalization'])

layers = vgg['layers']
print("layers.shape:",layers.shape)#layers.shape: (1, 43)[[1,2]]
layer = layers[0]
print("layer.shape:",layer.shape)#[1,2]layer.shape: (43,)

print("layer[0].shape:",layer[1].shape)#layer[0].shape: (1, 1)
a = layer[1]
print(a.dtype)
print(layer[0].dtype)

print("layer[0][0].shape:",layer[0][0].shape)#layer[0][0].shape: (1,)

print("layer[0][0][0].shape:",layer[0][0][0].shape)#layer[0][0][0].shape: ()

print("len(layer[0][0][0]):",len(layer[0][0][0]))#len(layer[0][0][0]): 5
#即weight(含有bias), pad(填充元素,无用), type, name, stride信息
count = 0
for i in range(43):

    #[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]
    if len(layer[i].dtype)==5:
        print(layer[i][0][0][3])
        count+=1
    elif len(layer[i].dtype)==2:
        #[('type', 'O'), ('name', 'O')]
        print(layer[i][0][0][1])
        count += 1
    elif len(layer[i].dtype) == 6:
        #[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]
        print(layer[i][0][0][0])
        count += 1
print(count)
def func():
    pass


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
# ['conv1_1']
# ['relu1_1']
# ['conv1_2']
# ['relu1_2']
# ['pool1']
# ['conv2_1']
# ['relu2_1']
# ['conv2_2']
# ['relu2_2']
# ['pool2']
# ['conv3_1']
# ['relu3_1']
# ['conv3_2']
# ['relu3_2']
# ['conv3_3']
# ['relu3_3']
# ['conv3_4']
# ['relu3_4']
# ['pool3']
# ['conv4_1']
# ['relu4_1']
# ['conv4_2']
# ['relu4_2']
# ['conv4_3']
# ['relu4_3']
# ['conv4_4']
# ['relu4_4']
# ['pool4']
# ['conv5_1']
# ['relu5_1']
# ['conv5_2']
# ['relu5_2']
# ['conv5_3']
# ['relu5_3']
# ['conv5_4']
# ['relu5_4']
# ['pool5']
# ['fc6']
# ['relu6']
# ['fc7']
# ['relu7']
# ['fc8']
# ['prob']
# 43