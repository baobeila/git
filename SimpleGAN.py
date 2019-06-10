#!/usr/bin/env python
# encoding: utf-8
"""
@version: JetBrains PyCharm 2017.3.2 x64
@author: baobeila
@contact: endoffight@gmail.com
@software: PyCharm
@file: SimpleGAN.py
@time: 2019/6/6 9:41
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
noise_size = 100
batchsize = 100
#tensorflow自带的方法read_data_sets()会先在指定文件夹中查看有无需要的文件，
# 若有，则直接加载，否则会从默认的网址自动下载
mnist = input_data.read_data_sets('D:\pycharm\Pycharmworkspace\mnist',one_hot=True)
X = tf.placeholder(tf.float32,shape=[None,784])
#判别器模型参数
Dw1 = tf.get_variable(shape=[784,128],initializer=tf.contrib.layers.xavier_initializer(),name='Dw1')
Db1 = tf.get_variable(shape=[128],initializer=tf.constant_initializer(0.0),name='Db1')
Dw2 = tf.get_variable(shape=[128,1],initializer=tf.contrib.layers.xavier_initializer(),name='Dw2')
Db2 = tf.get_variable(shape=[1],initializer=tf.constant_initializer(0.0),name='Db2')
thetaD = [Dw1,Db1 ,Dw2 ,Db2]
#生成器模型参数
Z = tf.placeholder(tf.float32,shape=[None,100])
Gw1 = tf.get_variable(shape=[100,128],initializer=tf.contrib.layers.xavier_initializer(),name='Gw1')
Gb1 = tf.get_variable(shape=[128],initializer=tf.constant_initializer(0.0),name='Gb1')
Gw2 = tf.get_variable(shape=[128,784],initializer=tf.contrib.layers.xavier_initializer(),name='Gw2')
Gb2 = tf.get_variable(shape=[784],initializer=tf.constant_initializer(0.0),name='Gb2')
thetaG = [Gw1,Gb1 ,Gw2 ,Gb2]

#判别器模型
def Discriminnator(X):
    D1 = tf.nn.relu(tf.matmul(X,Dw1 )+Db1)
    # D2 = tf.nn.sigmoid(tf.matmul(D1,Dw2 )+Db2)
    Dlogit = tf.matmul(D1, Dw2) + Db2
    return Dlogit
#生成器模型
def Generator(Z):
    G1 = tf.nn.relu(tf.matmul(Z,Gw1 )+Gb1)
    # G1 = tf.nn.leaky_relu(tf.matmul(Z,Gw1 )+Gb1)
    G2 = tf.nn.sigmoid(tf.matmul(G1,Gw2 )+Gb2)
    return G2
#噪声生成
batch_noise = np.random.uniform(-1., 1., size=[batchsize,noise_size])#batchsize
#输入数据
Gsample = Generator(Z)
Dlogitfake = Discriminnator(Gsample)
Dlogitreal = Discriminnator(X)
#计算损失
realloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dlogitreal,labels=tf.ones_like(Dlogitreal)))
fakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dlogitfake,labels=tf.zeros_like(Dlogitfake)))
Dloss = tf.add(realloss,fakeloss)
Gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dlogitfake,labels=tf.ones_like(Dlogitfake)))
D_opt = tf.train.AdamOptimizer(0.0005).minimize(Dloss,var_list=thetaD)
G_opt = tf.train.AdamOptimizer(0.0005).minimize(Gloss,var_list=thetaG)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())




    for itteration in range(10001):
        mb, _ = mnist.train.next_batch(batchsize)##batchsize
        sess.run(D_opt,feed_dict={X:mb,Z:batch_noise})
        sess.run(G_opt,feed_dict={Z:batch_noise})
        if itteration%1000 ==0:
            sample_noise = np.random.uniform(-1., 1., size=[4,noise_size])#一张图片里包含四张小图4*784
            g_out = sess.run(Gsample,feed_dict={Z:sample_noise})
            g_output = (g_out + 1) / 2#转到0-1之间
            fig = plt.figure(figsize=(2,2))#图像宽2英寸，高2英寸
            gs  = gridspec.GridSpec(2,2)#gridspec.GridSpec将整个图像窗口分成2行2列.
            gs.update(wspace = 0.015,hspace = 0.015)#子图之间的间距由wspace和hspace设置，边缘的距离不会发生改变
            for i,gout in enumerate(g_output):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')


                plt.imshow(gout.reshape([28, 28]), cmap='Greys_r')#颜色控制
            plt.savefig(r'D:\git\char\leakyrelu{}.png'.format(str(itteration).zfill(3)),bbox_inches='tight')#三位数显示
            #这样画布上的所有艺术家(包括图例)都适合保存的区域。如果需要，图形大小将自动调整。
            plt.close(fig)
        if  itteration%500 ==0:
            Dlosscur = sess.run(Dloss, feed_dict={X: mb, Z: batch_noise})
            Glosscur = sess.run(Gloss, feed_dict={Z: batch_noise})
            print('Iter:{}'.format(itteration))
            print('D loss:{:.4}'.format(Dlosscur))#:.4控制产生四位有效数字
            print('G loss:{:.4}'.format(Glosscur))
            # plt.show()
# def func():
#     pass
#
#
# class Main():
#     def __init__(self):
#         pass
#
#
# if __name__ == '__main__':
#     pass