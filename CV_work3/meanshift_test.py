# -*- coding: utf-8 -*- 
# @Time : 2020/3/28 18:24 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : meanshift_test.py
# @Software: PyCharm
'''
与opencv官方文档实例相同
'''
import numpy as np
import cv2

cap = cv2.VideoCapture('test1.avi')

ret, frame = cap.read()

r, h, c, w =250, 250, 300, 200
track_window = (c, r, w, h)

roi = frame[r: r + h, c: c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 2556., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img1 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img1', img1)

        k = cv2.waitKey(0x110000)
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img1)

    else:
        break

cv2.destroyAllWindows()
cap.release()




