#coding=utf-8
#Version:python3.5.2
"""创建TXT类型的文件，描述数据集"""
import os
dictag = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}
def createtxtfile(filedir,targetdir):
    """Args:
            filedir:图片文件的上一级目录
        Return：
        filename:图片文件的绝对路径  Cr_1.bmp
        label:每张图片的标签"""
    f = open(targetdir, 'w')
    for filenames in os.listdir(filedir):
        filename = os.path.join(filedir, filenames)
        labels1 = filename.split('\\')[-1]
        labels = labels1.split('_')[0]
        label = dictag[labels]
        f.write(str(filename)+' '+str(label)+'\n')
    f.close()
if __name__ == "__main__":
    # filedir =r'D:\2345Downloads\train'
    # createtxtfile(filedir, 'train.txt')
    filedir =r'D:\2345Downloads\test'
    createtxtfile(filedir,'val.txt')