# CCNU_CV
## CCNU CV Course Work1  
**作业说明**  

找一段采用固定摄像头拍摄的视频，写一个小程序，使用鼠标指定坐标，统计该坐标上的背景像素点的概率分布。

要求提交：  

（1）源代码（一附件提交。）；     

（2）至少一个坐标上的背景像素点的概率分布图。  



**工程文件结构说明**    

+ **beta1.0&emsp; version 1**  
    + test1.avi &emsp;测试视频1  
    + test2.avi &emsp;测试视频2   
    + test3.avi &emsp;测试视频3    
    + test.4avi &emsp;测试视频4  
      - 文件夹/test1  &emsp;获取视频1的帧图像  
      - 文件夹/test2  &emsp;获取视频2的帧图像  
      - 文件夹/test3  &emsp;获取视频3的帧图像  
      - 文件夹/test4  &emsp;获取视频4的帧图像  
    + BGImage.py  &emsp;高斯混合模型分割背景  
    + get_scr.py  &emsp;获得帧图片  
    + click.py  &emsp;点击图像获取坐标  
    + main.py  &emsp;获得RGB分布直方图  
      

版本一实现了视频帧图片的获取，然后实现任取背景一点可生成该点RGB概率分布直方图   

/********************************/
+ **beta2.0&emsp; version 2**  
    - main.py &emsp;主功能函数
    - 文件夹/test1  &emsp;获取视频1的帧图像  
    - 文件夹/test2  &emsp;获取视频2的帧图像  
    - 文件夹/test3  &emsp;获取视频3的帧图像  
    - 文件夹/test4  &emsp;获取视频4的帧图像  

运行主程序，输入待分析视频（无需添加视频后缀），双击图片获取对应坐标的rgb分布

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/1.png)    

程序运行中会截取视频帧图片，然后保存在相应的文件夹中 

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/2.png)     

帧图片读取保存后，会播放原视频，并可在视频结束后点击视频中任意一点获得坐标；  
然后进行接下来的视频分析    

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/3.png)    

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/4.png)    

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/5.png)   

如图所示，点击视频中任意一点获得该点的RGB频率分布直方图    

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/6.png)    

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/7.png)     

 

## CCNU CV Course Work2   
**作业说明**  

自拍一张照片，并使用OpenCv进行人脸检测。  



**工程文件说明**

+ **image_detector**  &emsp;图像人脸识别  
    + image_detector.py  &emsp;功能主函数   
        + test1.jpg  &emsp;测试图片1  
        + test2.jpg  &emsp;测试图片2  
        + output.jpg  &emsp;输出检测后的图片    
        + haarcascade_eye.xml  &emsp;  
        + haarcascade_frontalface_default.xml  &emsp;  
        

/********************************/  
+ **video_detector** &emsp;视频人脸识别  
    + video_detector.py  &emsp;功能主函数   
        + result.mp4  &emsp;检测结果录屏  
        + haarcascade_frontalface_alt2.xml  &emsp;  
        + haarcascade_frontalface_default.xml  &emsp;  
        
        

图像人脸检测结果：  
![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work2/instruction/1.jpg)  

视频人脸检测结果：  
![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work2/instruction/2.png)



##  CCNU CV Course Work3  

**作业说明**   

用OpenCv写出MeanShift目标跟踪算法，并跟踪自己的脸  



**工程文件说明**   

+ meanshift_test.py   &emsp;meanshift算法进行人脸检测   
+ camshift_test.py  &emsp; camshift算法进行人脸检测
+ test1.avi    &emsp;测试视频   
+ screencap.mp4    &emsp;录屏  
+ 0.jpg   &emsp; 随机截取帧图片       





meanshift_test.py&&camshift_test.py 运行后  按<u>Ctrl + 任意字母键or数字键</u>可以逐帧观察视频中帧图片中的检测结果

<u>Ctrl+s即保存该帧图片</u>（单独字母键or数字键即自动保存当前帧图片为该字母or数字名称的jpg文件，可覆盖） 





**MeanShift分析**  

MeanShift算法用于视频目标跟踪时，其实就是采用目标的颜色直方图作为搜索特征，通过不断迭代meanShift向量使得算法收敛于目标的真实位置，从而达到跟踪的目的。  

+ <u>在目标跟踪中：Meanshift有以下几个优势：</u>

  + 算法计算量不大，在目标区域已知的情况下完全可以做到实时跟踪  
  + 采用核函数直方图模型，对边缘遮挡、目标旋转、变形和背景运动不敏感  

+ <u>同时，MeanShift算法也存在着以下一些缺点：</u>  

  + 缺乏必要的模板更新；  
  + 跟踪过程中由于窗口宽度大小保持不变，当目标尺度有所变化时，跟踪就会失败；  
  + 当目标速度较快时，跟踪效果不好；  
  + 直方图特征在目标颜色特征描述方面略显匮乏，缺少空间信息；    

+ <u>由于其计算速度快，对目标变形和遮挡有一定的鲁棒性，其中一些在工程实际中也可以对其作出一些改进和调整如下：</u>

  + 引入一定的目标位置变化的预测机制，从而更进一步减少MeanShift跟踪的搜索时间，降低计算量；  
  + 可以采用一定的方式来增加用于目标匹配的“特征”；  
  + 将传统MeanShift算法中的核函数固定带宽改为动态变化的带宽；  
  + 采用一定的方式对整体模板进行学习和更新；    

  
  
    
  
  

**CamShift分析**   

CamShift算法，全称是 Continuously AdaptiveMeanShift，顾名思义，它是对Mean Shift 算法的改进，能够自动调节搜索窗口大小来适应目标的大小，可以跟踪视频中尺寸变化的目标。它也是一种半自动跟踪算法，需要手动标定跟踪目标。基本思想是以视频图像中运动物体的颜色信息作为特征，对输入图像的每一帧分别作 Mean-Shift 运算，并将上一帧的目标中心和搜索窗口大小(核函数带宽)作为下一帧 Mean shift 算法的中心和搜索窗口大小的初始值，如此迭代下去，就可以实现对目标的跟踪。因为在每次搜索前将搜索窗口的位置和大小设置为运动目标当前中心的位置和大小，而运动目标通常在这区域附近，缩短了搜索时间；另外，在目标运动过程中，颜色变化不大，故该算法具有良好的鲁棒性。

+ <u>与MeanShift相比之下的优点：</u>
  + 算法改进了 Mean-Shift 跟踪算法的缺陷，在跟踪过程中能够依据目标的尺寸调节搜索窗口大小，有尺寸变化的目标可准确定位。
  + 对系统资源要求不高，时间复杂度低，在简单背景下能够取得良好的跟踪效果
+ <u>存在的缺陷：</u>
  + CamShfit 算法在计算目标模板直方图分布时，没有使用核函数进行加权处理，也就是说目标区域内的每个像素点在目标模型中有着相同的权重，故 CamShfit 算法的抗噪能力低于Mean-Shift跟踪算法。
  + CamShift 算法中没有定义候选目标，直接利用目标模板进行跟踪。
  + CamShift 算法采用 HSV 色彩空间的H分量建立目标直方图模型，仍然只是依据目标的色彩信息来进行跟踪，当目标与背景颜色接近或者被其他物体遮挡时，CamShift 会自动将其包括在内，导致跟踪窗口扩大，有时甚至会将跟踪窗口扩大到整个视频大小，导致目标定位的不准确，连续跟踪下去造成目标的丢失。  

  



**MeanShift目标跟踪结果：**  

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work3/0.jpg)   



**camShift目标跟踪结果：**

![image](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work3/camshift.jpg)   



<u>【注】：readme插图用的是网络插图法（比较依赖网络通常），如果不能正常浏览，进入instruction文件夹即可正常浏览插图。</u>    