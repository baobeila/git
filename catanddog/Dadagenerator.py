#coding=utf-8
#Version:python3.5.2
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import convert_to_tensor
from tensorflow.python.framework import ops
class Datagenerator(object):
    def __init__(self,txtfile,mode,batchsize,num,shuffle,buffer_size=1000):
        self.txtfile = txtfile
        self.readtxt()
        self.mode = mode
        self.data_size = len(self.imagelabel)
        self.num = num
        if shuffle:
            self.shuffle_list()#对路径打乱
        data = self.Data()

        if shuffle:

            data =data.shuffle(buffer_size=buffer_size)#什么意思在缓冲区大小对路径打乱
        data = data.batch(batchsize)
        self.data = data
    def Data(self):


        self.imagepath = ops.convert_to_tensor(self.imagepath, dtype=dtypes.string)
        self.imagelabel = ops.convert_to_tensor(self.imagelabel, dtype=dtypes.int32)
        data = tf.data.Dataset.from_tensor_slices((self.imagepath,self.imagelabel))
        if self.mode =='training':
            data = data.map(self.parsetrain)
        elif self.mode =='inference':
            data = data.map(self.parseval)#data数据处理完成，后打乱整理
        return data
        # return self.imagepath ,self.imagelabel

    def parsetrain(self,imagepath,imagelabel):#此时前面不用加self
        imgstring = tf.read_file(imagepath)
        deco_image = tf.image.decode_jpeg(imgstring,channels=3)
        imgdata = tf.image.convert_image_dtype(deco_image,dtype=tf.float32)
        im_resized= tf.image.resize_images(imgdata,[227,227],method=0)
        """进行数据增强"""
        one_hot = tf.one_hot(imagelabel,self.num)
        return im_resized,one_hot

    def parseval(self,imagepath,imagelabel):
        imgstring = tf.read_file(imagepath)
        deco_image = tf.image.decode_jpeg(imgstring,channels=3)
        imgdata = tf.image.convert_image_dtype(deco_image, dtype=tf.float32)
        im_resized = tf.image.resize_images(imgdata, [227, 227], method=0)
        one_hot = tf.one_hot(imagelabel, self.num)
        return im_resized, one_hot
    def readtxt(self):
        #self.xxx是全局的
        self.imagepath = []
        self.imagelabel = []
        with open(self.txtfile) as f:
            for line in f:
                lines = line.rstrip('\n')
                lines = lines.split(' ')
                self.imagepath.append(lines[0])
                self.imagelabel.append(int(lines[1]))#先将字符转为整数

        return self.imagepath,self.imagelabel

    def shuffle_list(self):
        path = self.imagepath
        labels = self.imagelabel
        permutation = np.random.permutation(len(labels))  # 随机打散训练数据,同时保持训练数据与标签的对齐.
        self.imagepath = []
        self.imagelabel = []
        for i in permutation:
            self.imagepath.append(path[i])
            self.imagelabel.append(labels[i])
if __name__ == '__main__':
  main = Datagenerator(r'D:\pycharm\train.txt','training',2,1,100,32)
  la = main.data
  print(la.output_types)
  print(la.output_shapes)
  print(la)
  # print(len(im))