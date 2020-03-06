# -*- coding: utf-8 -*- 
# @Time : 2020/3/6 15:25 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : image_detector.py
# @Software: PyCharm
# -*-coding:utf8-*-
# !/usr/bin/env python

import cv2
import numpy as np
from skimage import io

img = cv2.imread("test1.jpg")
color = (0, 255, 0)


grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
classfier = cv2.CascadeClassifier("D:\\github_baobaotql\\CCNU_CV\\CV_work2\\image_detector\\haarcascade_frontalface_default.xml")
faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

# 大于0则检测到人脸
if len(faceRects) > 0:
    # 单独框出每一张人脸
    for faceRect in faceRects:
     x, y, w, h = faceRect
     cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)

faces = classfier.detectMultiScale(grey,1.2,2)
print("检测到人脸 ： ",len(faces))

# 写入图像
cv2.imwrite('output.jpg',img)
cv2.imshow("Find Faces!",img)
cv2.waitKey(0)


