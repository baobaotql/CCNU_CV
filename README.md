# CCNU_CV
## CCNU CV Course Work1  
**工程文件结构说明  
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

运行主程序，输入待分析视频（无需添加视频后缀）

![avatar](https://github.com/baobaotql/CCNU_CV/blob/master/CV_work1/instruction/1.png)    

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

注：readme插图用的是网络插图法（比较依赖网络），如果不能正常浏览，进入instruction文件夹即可正常浏览。
