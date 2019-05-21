#coding=utf-8
#Version:python3.5.2
"""创建TXT类型的文件，描述数据集"""
import os
dictag = {'cat': 0, 'dog': 1}
def createtxtfile(filedir,targetdir):
    """Args:
            filedir:图片文件的上一级目录
        Return：
        filename:图片文件的绝对路径  验证集和测试集图片命名格式cat.202.jpg
        label:每张图片的标签 cat为0，dog为1
        targetdir:txt文件名"""
    f = open(targetdir, 'w')
    for filenames in os.listdir(filedir):
        if not filenames.endswith(".jpg") or filenames.startswith('.'):
            continue
        filename = os.path.join(filedir, filenames)
        labels1 = filename.split('\\')[-1]
        labels = labels1.split('.')[0]
        label = dictag[labels]
        f.write(str(filename)+' '+str(label)+'\n')
    f.close()
if __name__ == "__main__":

    # filedir =r'D:\pycharm\catdogval'
    # createtxtfile(filedir,r'D:\pycharm\val.txt')
    filedir =r'D:\pycharm\catdogtrain'
    createtxtfile(filedir,r'D:\pycharm\train.txt')
