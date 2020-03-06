# -*- coding: utf-8 -*- 
# @Time : 2020/3/2 23:03 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : main.py 
# @Software: PyCharm


import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def draw_hist(list_b, list_g, list_r):
    plt.subplot(311)
    plt.hist(list_b, 256, [0, 256], label='RGB—B', color='b')
    plt.subplot(312)
    plt.hist(list_g, 256, [0, 256], label='RGB—G', color='g')
    plt.subplot(313)
    plt.hist(list_r, 256, [0, 256], label='RGB—R', color='r')
    plt.show()
    color = ('b', 'g', 'r')

def click(event, x, y, flags, param):
    list_b = []
    list_g = []
    list_r = []
    if event == cv.EVENT_LBUTTONDBLCLK:  # 左键双击
        height = images[0].shape[0]
        width = images[0].shape[1]
        print('窗口大小：（%d,%d)' % (width, height))
        print('坐标点位置：（%d,%d)' % (x, y))
        for each in param:
            list_b.append(each[y, x, 0])
            list_g.append(each[y, x, 1])
            list_r.append(each[y, x, 2])
    else:
        return
    draw_hist(list_b, list_g, list_r)


if __name__ == "__main__":
    video_based_path = 'D:/github_baobaotql/CCNU_CV/CV_work1/beta1.0/'
    print('输入待分析视频名称（无文件后缀）：')
    video_name = input()
    video_add_path = video_based_path + video_name + '.avi'

    #print('获取帧图片的视频', video_name)
    times = 0
    # 提取每一帧图片
    frameFrequency = 1
    outPutDirName = video_name + '/'

    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    camera = cv.VideoCapture(video_add_path)
    while True:
        times += 1
        res, scr_image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            cv.imwrite(outPutDirName + str(times) + '.jpg', scr_image)
            #测试打印
            print(outPutDirName + str(times) + '.jpg')
    print('帧图片提取结束')

    camera = cv.VideoCapture(video_add_path)
    times = 0
    images = []
    while True:
        times += 1
        res, image = camera.read()
        #最后一帧
        if not res:
            break
        images.append(image)
        cv.imshow('video', image)
        cv.waitKey(10)

    camera.release()
    cv.destroyAllWindows()

    cv.imshow('video', images[-1])
    cv.namedWindow("video")
    # 鼠标触发回调事件
    cv.setMouseCallback('video', click, images)
    cv.imshow('video', images[-1])

    while True:
        try:
            cv.waitKey(100)
        except Exception as e:
            cv.destroyAllWindows()
            break


