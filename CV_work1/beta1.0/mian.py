# -*- coding: utf-8 -*- 
# @Time : 2020/2/28 20:13 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : mian.py
# @Software: PyCharm

import os
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

path = '../CV_work1/beta1.0/test3/'

#读取path文件夹下所有文件的名字
bgimagelist = os.listdir(path)

#测试输出读取的图片
#print(bgimagelist)

list_b = []
list_g = []
list_r = []
for imgname in bgimagelist:

    if (imgname.endswith(".jpg")):
        #测试
        #print(imgname)
        #测试并检查图片
        img = cv2.imread('../CV_work1/beta1.0/test3/'+imgname)
        #print(img.shape)

        value_b = img[160,130,0]
        value_g = img[160,130,1]
        value_r = img[160,130,2]
        list_b.append(value_b)
        list_g.append(value_g)
        list_r.append(value_r)
#打印测试
print(list_b)
print(list_g)
print(list_r)

# def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
#     '''
#
#     :param myList:  list
#     :param Title:   抬头
#     :param Xlabel:  X轴
#     :param Ylabel:  Y轴
#     :param Xmin:    XY轴的范围
#     :param Xmax:
#     :param Ymin:
#     :param Ymax:
#     :return:
#     '''
#     plt.hist(myList,100)
#     plt.xlabel(Xlabel)
#     plt.xlim(Xmin,Xmax)
#     plt.ylabel(Ylabel)
#     plt.ylim(Ymin,Ymax)
#     plt.title(Title)
#     plt.show()
def draw_hist(list_b, list_g, list_r):
    plt.subplot(311)
    plt.hist(list_b, 256, [0, 256], label='RGB—B', color='b')
    plt.subplot(312)
    plt.hist(list_g, 256, [0, 256], label='RGB—G', color='g')
    plt.subplot(313)
    plt.hist(list_r, 256, [0, 256], label='RGB—R', color='r')
    plt.show()
draw_hist(list_b, list_g, list_r)









