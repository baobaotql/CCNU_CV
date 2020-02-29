# -*- coding: utf-8 -*- 
# @Time : 2020/2/28 20:13 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : Get_RGB.py 
# @Software: PyCharm


import os
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

path = '../CV_work1/test3/'

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
        img = cv2.imread('../CV_work1/test3/'+imgname)
        #print(img.shape)

        value_b = img[160,130,0]
        value_g = img[160,130,1]
        value_r = img[160,130,2]
        list_b.append(value_b)
        list_g.append(value_g)
        list_r.append(value_r)
print(list_b)
print(list_g)
print(list_r)

def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    '''

    :param myList:  list
    :param Title:   抬头
    :param Xlabel:  X轴
    :param Ylabel:  Y轴
    :param Xmin:    XY轴的范围
    :param Xmax:
    :param Ymin:
    :param Ymax:
    :return:
    '''
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

#频率直方图
draw_hist(list_b,'RGB_Blue','Area','Probability',0,255,0.0,500)
draw_hist(list_g,'RGB_Green','Area','Probability',0,255,0.0,500)
draw_hist(list_r,'RGB_Red','Area','Probability',0,255,0.0,500)

#概率分布直方图
n, bins, patches = plt.hist(list_b,1000, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins, 100, 15)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'RGB_Blue')
plt.subplots_adjust(left=0.15)
plt.show()

n, bins, patches = plt.hist(list_g,1000, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins, 100, 15)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'RGB_Green')
plt.show()

n, bins, patches = plt.hist(list_r,1000, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins, 100, 15)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'RGB_Red')
plt.show()
# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
# #第二个参数是柱子宽一些还是窄一些，越大越窄越密
# ax0.hist(list_b,40,normed=1,histtype='bar',facecolor='blue',alpha=0.75)
# ##pdf概率分布图，一万个数落在某个区间内的数有多少个
# ax0.set_title('RGB_Blue')
# ax1.hist(list_b,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
# #cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
# ax1.set_title("cdf")
# fig.subplots_adjust(hspace=0.4)
# plt.show()








