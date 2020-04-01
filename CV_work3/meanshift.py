# -*- coding: utf-8 -*- 
# @Time : 2020/3/28 18:24 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : meanshift.py
# @Software: PyCharm

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 将帧图片保存为视频
def save_video(video, fps, save_filename='../output.avi'):
    fourcc = cv.VideoWriter_fourcc(*'avi')
    img = video[0]
    video_writer = cv.VideoWriter(save_filename, fourcc, fps, (img.shape[1], video.shape[0]), 1)
    for img in video:
        video_writer.write(cv.convertScaleAbs(img))


#  核函数加权下的直方图分布
def cal_histogram_kernel(area):
    width = area.shape[1]
    height = area.shape[0]
    # 初始化kernel权重
    weight = kernel_weight(height, width)
    weight_sum = np.sum(weight)
    # rgb颜色空间量化为16 * 16 * 16bins
    histogram = np.zeros(16*16*16)
    for i in range(0, height):
        for j in range(0, width):
            r_feature = int(area[i, j, 0] / 16)
            g_feature = int(area[i, j, 1] / 16)
            b_feature = int(area[i, j, 2] / 16)
            histogram[r_feature*256+g_feature*16+b_feature] += weight[i, j]
    return histogram/weight_sum


def kernel_weight(row, col):
    # Epanechnikov 核
    weight = np.zeros((row, col))
    middle_x = row/2
    middle_y = col/2
    radius_square = np.square(middle_x) + np.square(middle_y)
    for i in range(0, row):
        for j in range(0, col):
            dist = np.square(i-middle_x) + np.square(j-middle_y)
            weight[i, j] = 1-(dist/radius_square)
    return weight


# 目标检测
def object_detect(img):
    face_cascade = \
        cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(5, 5))
    return face[0]


def draw_rectangle(img, x_start, y_start, width, height):
    img = cv.rectangle(img, (x_start, y_start), (x_start+width, y_start + height), (255, 0, 0), 2)
    return img


def mean_shift(model_feature, img, x_ini, y_ini, width, height):
    x_start = x_ini
    y_start = y_ini
    iteration = 0
    while iteration < 20:
        # 候选区域
        detect_area = img[y_start:y_start+height, x_start:x_start+width]
        detect_feature = cal_histogram_kernel(detect_area)
        weight = np.true_divide(model_feature, detect_feature)
        weight = np.nan_to_num(weight)
        weight = np.sqrt(weight)

        x_move = 0
        y_move = 0
        weight_sum = 0
        for i in range(0, height):
            for j in range(0, width):
                try:
                    r_feature = int(detect_area[i, j, 0] / 16)
                except Exception as e:
                    print("error")
                    break
                r_feature = int(detect_area[i, j, 0] / 16)
                g_feature = int(detect_area[i, j, 1] / 16)
                b_feature = int(detect_area[i, j, 2] / 16)
                w_i = weight[r_feature*256+g_feature*16+b_feature]
                x_move += w_i*(j-width/2)
                y_move += w_i*(i-height/2)
                weight_sum += w_i
        x_start += int(x_move/weight_sum)
        y_start += int(y_move/weight_sum)
        # 若矩形框想要越界
        if x_start+width >= img.shape[1]:
            print('迭代次数：%d' % iteration)
            if y_start+height >= img.shape[0]:
                return img.shape[1]-width-1, img.shape[0]-height-1
            else:
                return img.shape[1] - width - 1, y_start
        # y纵向越界，x不越界
        if y_start + height >= img.shape[0]:
            print('迭代次数：%d' % iteration)
            return x_start, img.shape[0] - height - 1

        if (np.abs(int(x_move/weight_sum)) < 1) and (np.abs(int(y_move/weight_sum)) < 1):
            break
        iteration += 1
    print('迭代次数：%d' % iteration)
    return x_start, y_start


if __name__ == "__main__":
    video_path = 'test1.avi'
    camera = cv.VideoCapture(video_path)
    # 码率
    fps = int(camera.get(cv.CAP_PROP_FPS))
    _, image = camera.read()
    x, y, w, h = object_detect(image)
    img_rectangle = draw_rectangle(image, x, y, w, h)
    cv.imshow('rectangle', img_rectangle)
    cv.waitKey(10)
    model = image[y:y + h, x:x + w, :]
    images = []
    # 目标模板的概率直方图
    q_u = cal_histogram_kernel(model)
    # plt.plot(range(0, 4096), q_u, label='model', color='b')
    # plt.show()
    images.append(img_rectangle)
    times = 1
    while True:
        times += 1
        res, image = camera.read()
        print('第%d帧图片' % times)
        x, y = mean_shift(q_u, image, x, y, w, h)
        img_rectangle = draw_rectangle(image, x, y, w, h)
        images.append(img_rectangle)
        # 最后一帧
        if not res:
            break
        cv.imshow('FaceTracing', img_rectangle)
        cv.waitKey(10)
    save_video(images, int(camera.get(cv.CAP_PROP_FPS)))
    camera.release()
    cv.destroyAllWindows()


