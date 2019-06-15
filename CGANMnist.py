#!/usr/bin/env python
# encoding: utf-8
"""
@version: JetBrains PyCharm 2017.3.2 x64
@author: baobeila
@contact: endoffight@gmail.com
@software: PyCharm
@file: CGANMnist.py
@time: 2019/6/14 9:26
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\pycharm\Pycharmworkspace\mnist',one_hot=True)
real_imagesize = mnist.train.images.shape[1]#784
noise_size = 100#噪声维度取100
digitnum = 10
smooth = 0.05#做损失的光滑？
learning_rate = 0.0001
epochs = 210
batch_size = 100
k=10
#创建logs文件目录
if not os.path.exists('logdir'):
    os.makedirs('logdir')
real_image = tf.placeholder(tf.float32,shape=[None,real_imagesize],name='real_image')#D的输入
real_label = tf.placeholder(tf.float32,shape=[None,digitnum],name='real_image_digit')#D的输入，G的输入,转为one-hot编码
noise_image = tf.placeholder(tf.float32,shape=[None,noise_size],name='noise_image')
#生成器和判别器的主要构架为全连接网络
def FClayer(name,value,output_node):
    with tf.variable_scope(name,reuse=None) as scope:
        prenode = value.get_shape().as_list()
        W = tf.get_variable("W",shape=[prenode[1],output_node],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',shape=[output_node],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        return tf.matmul(value,W)+b#未经激活的
#噪声生成方式
def get_noise(batch_size,noise_class='uniform'):
    if noise_class=='uniform':
        noise = np.random.uniform(-1,1,size=(batch_size,noise_size))
    elif noise_class =='normal':
        noise = np.random.normal(-1,1,size=(batch_size,noise_size))
    return noise.astype('float32')
def Generator(noise_image,real_label,reuse = False):
    with tf.variable_scope('Generator',reuse=reuse):
        catimg_dig = tf.concat([real_label,noise_image],1)#需要看下维度
        output = FClayer(name='GFC1',value=catimg_dig,output_node=128)
        output = tf.nn.leaky_relu(output,alpha=0.01,name='leaky_relu1')
        output = tf.layers.dropout(output,rate=0.5)

        output = FClayer(name='GFC2',value=output,output_node=128)
        output = tf.nn.leaky_relu(output,alpha=0.01,name='leaky_relu2')
        output = tf.layers.dropout(output,rate=0.5)

        logit = FClayer(name='GFC3',value=output,output_node=784)
        outputs = tf.nn.tanh(logit)
        return logit,outputs


def Discriminator(image,real_label,reuse = False):
    with tf.variable_scope('Discriminator',reuse=reuse):
        catimg_dig = tf.concat([real_label,image],1)#需要看下维度
        output = FClayer(name='DFC1',value=catimg_dig,output_node=128)
        output = tf.nn.leaky_relu(output,alpha=0.01,name='leaky_relu1')
        output = tf.layers.dropout(output,rate=0.5)

        output = FClayer(name='DFC2',value=output,output_node=128)
        output = tf.nn.leaky_relu(output,alpha=0.01,name='leaky_relu2')
        output = tf.layers.dropout(output,rate=0.5)

        logit = FClayer(name='DFC3',value=output,output_node=1)
        # outputs = tf.nn.tanh(logit)
        outputs = tf.nn.sigmoid(logit)#用0到1进行衡量
        return logit,outputs
#生成器
g_logit,g_outputs = Generator(noise_image=noise_image,real_label=real_label)
d_logitreal,d_outputsreal=Discriminator(image=real_image ,real_label=real_label)#realimage转为-1到1
d_logitfake,d_outputsfake=Discriminator(image=g_outputs ,real_label=real_label,reuse=True)#这里的变量复用原因
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logitreal,
                                                                         labels = tf.ones_like(d_logitreal)) * (1 - smooth))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logitfake,
                                                                         labels = tf.zeros_like(d_logitfake)))
# 判别器loss
d_loss = tf.add(d_loss_real, d_loss_fake)
#生成器loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logitfake,
                                                                         labels = tf.ones_like(d_logitfake)) * (1 - smooth))

#找到训练的变量
train_vars = tf.trainable_variables()
#生成器中要训练更新的变量
g_vars = [var for var in train_vars if var.name.startswith("Generator")]
#判别器中要训练更新的变量
d_vars = [var for var in train_vars if var.name.startswith("Discriminator")]
#优化操作
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

def train():
    # 保存loss值
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in xrange(epochs):#对所有训练数据迭代的轮数
            for i in xrange(mnist.train.num_examples // (batch_size * k)):#k用来控制多少个batchsize做一次写入
                for j in xrange(k):
                    batch = mnist.train.next_batch(batch_size)
                    digits = batch[1]#得到标签100*10
                    images = batch[0].reshape((batch_size, 784))
                    images = 2 * images - 1#转为-1到1之间
                    # generator input noises
                    noises = get_noise( batch_size=batch_size )#提供默认类型
                    sess.run([d_train_opt, g_train_opt],
                             feed_dict={real_image : images, noise_image: noises, real_label : digits})
            fd_loss,fd_loss_real,fd_loss_fake,fg_loss=sess.run([d_loss,d_loss_real ,d_loss_fake,g_loss],
             feed_dict={real_image: images, noise_image: noises, real_label: digits})
            print("Epoch {}/{}".format(e + 1, epochs),
                  "Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(
                      fd_loss, fd_loss_real , fd_loss_fake),
                  "Generator loss: {:.4f}".format(fg_loss))
            #查看每轮结果
            noises = get_noise(batch_size=batch_size)
            label = np.array([0,0,0,0,0,0,0,0,1,0],dtype='float32')
            labels = np.tile(label,(100,1))
            _,gen = sess.run(Generator(noise_image=noise_image,real_label=real_label,reuse=True),feed_dict={noise_image:noises,real_label:labels})
            # print(gen.shape)画图程序将100*784以图的形式画出来
            gen = gen.reshape(-1,28,28)#shape (100,28,28)  -1,1
            gen = (gen+1)/2
            plotImages(gen,e)
def plotImages(gen,e):
    r,c = 10,10
    fig,axs = plt.subplots(r,c)#与SimpleGAN比较体会不同画网格图的代码风格
    index = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen[0],cmap='Greys_r')
            axs[i, j].axis('off')
            index += 1
    if not os.path.exists('gen_mnist1'):
        os.makedirs('gen_mnist1')
    fig.savefig('gen_mnist1/%d.jpg' % e)
    plt.close()





class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    train()