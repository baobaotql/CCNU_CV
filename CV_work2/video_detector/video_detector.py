# -*- coding: utf-8 -*- 
# @Time : 2020/3/6 16:50 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : video_detector.py.py 
# @Software: PyCharm

import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    xmlfile = 'D:\\github_baobaotql\\CCNU_CV\\CV_work2\\video_detector\\haarcascade_frontalface_alt2.xml'

    face_cascade = cv2.CascadeClassifier(xmlfile)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.15, minNeighbors=5,minSize=(5, 5),)
    print("发现{0}个目标!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)
    cv2.imshow("video_detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

