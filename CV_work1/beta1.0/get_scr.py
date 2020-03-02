# -*- coding: utf-8 -*- 
# @Time : 2020/2/28 19:05 
# @Author : BaoBao
# @Mail : baobaotql@163.com 
# @File : get_scr.py
# @Software: PyCharm

#获得视频多帧截图
import os
import cv2 as cv

print('输入要获取帧图片的视频：')
sourceFileName=input()
print('获取帧图片的视频',sourceFileName)
video_path = os.path.join("", "", sourceFileName+'.avi')
times=0
#提取每一帧图片
frameFrequency=1
outPutDirName= sourceFileName+'/'

if not os.path.exists(outPutDirName):
    os.makedirs(outPutDirName)
camera = cv.VideoCapture(video_path)
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        cv.imwrite(outPutDirName + str(times)+'.jpg', image)
        #测试打印
        print(outPutDirName + str(times)+'.jpg')
print('图片提取结束')
camera.release()


'''
videos_src_path = "D:/github_baobaotql/CCNU_CV/CV_work1/"
video_formats = [".avi"]
frames_save_path = "D:/github_baobaotql/CCNU_CV/CV_work1/"
width = 1920
height = 1080
time_interval = 0.5


def video2frame(video_src_path, formats, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    videos = os.listdir(video_src_path)

    def filter_format(x, all_formats):
        if x[-4:] in all_formats:
            return True
        else:
            return False

    videos = filter(lambda x: filter_format(x, formats), videos)

    for each_video in videos:
        print("正在读取视频：", each_video)

        each_video_name = each_video[:-4]
        os.mkdir(frame_save_path + each_video_name)
        each_video_save_full_path = os.path.join(frame_save_path, each_video_name) + "/"

        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("读取失败!")

        while (success):
            success, frame = cap.read()
            print
            "---> 正在读取第%d帧:" % frame_index, success

            if frame_index % interval == 0:
                resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
                cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, resize_frame)
                frame_count += 1

            frame_index += 1

    cap.release()


if __name__ == '__main__':
    video2frame(videos_src_path, video_formats, frames_save_path, width, height, time_interval)

'''
