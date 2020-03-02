# -*- coding: utf-8 -*- 
# @Time : 2020/2/28 16:14 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : BGImage.py
# @Software: PyCharm

#功能测试：MOG获取背景
import cv2 as cv

capture = cv.VideoCapture("../CV_work1/beta1.0/test3.avi")
mog = cv.createBackgroundSubtractorMOG2()
se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
while True:
    ret, image = capture.read()
    if ret is True:
        fgmask = mog.apply(image)
        ret, binary = cv.threshold(fgmask, 220, 255, cv.THRESH_BINARY)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
        
        bgimage = mog.getBackgroundImage()
        #save backgroud image
        cv.imwrite("bgimage.jpg",bgimage)

        cv.imshow("bgimage", bgimage)
        cv.imshow("frame", image)
        cv.imshow("fgmask", binary)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()
