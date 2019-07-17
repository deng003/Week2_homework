import cv2
import numpy as np

img = cv2.imread('D:\Artificial_Intelligence\pic\AGJR7725.JPG',0)

# Gaussian Kernel Effect
g_img1 = cv2.GaussianBlur(img,(3,3),5)      #（4，4） 为高斯核的大小，核越大，作用越广，图形越模糊
g_img2 = cv2.GaussianBlur(img,(17,17),5)    #5为标准差 标准差越大，分布越大，图像越模糊
g_img3 = cv2.GaussianBlur(img,(3,3),1)

kernel_lap1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)       #会产生双边缘效果
kernel_lap2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)   #会产生强烈的双边缘效果
kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)            #产生单边缘效果
kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)            #产生强烈的单边缘效果
kernel3 = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)            #图形模糊，只是中心像素比例较大
lap_img1 = cv2.filter2D(img, -1, kernel=kernel_lap1)
lap_img2 = cv2.filter2D(img, -1, kernel=kernel_lap2)
img11 = cv2.filter2D(img, -1, kernel=kernel1)
img12 = cv2.filter2D(img, -1, kernel=kernel2)
img13 = cv2.filter2D(img, -1, kernel=kernel3)

kernel21 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)         #图形锐化
kernel22 = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)            #图形模糊，只是中心像素比例较大
kernel23 = np.array([[0, 1, 0], [1, -5, 1], [0, 1, 0]], np.float32)            #单边缘效果，灰度值更低
img21 = cv2.filter2D(img, -1, kernel=kernel21)
img22 = cv2.filter2D(img, -1, kernel=kernel22)
img23 = cv2.filter2D(img, -1, kernel=kernel23)
#cv2.imshow('img21', img21)
#cv2.imshow('img22', img22)
#cv2.imshow('img23', img23)
#key = cv2.waitKey()

edgex = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
edgey = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sharpx_img = cv2.filter2D(img, -1, kernel=edgex)            # 显示X轴单边缘
sharpy_img = cv2.filter2D(img, -1, kernel=edgey)            # 显示y轴单边缘
sharpxy_img = cv2.filter2D(img, -1, kernel=edgex*edgey)     # 显示双边缘
#cv2.imshow('sharpx_img', sharpx_img)
#cv2.imshow('sharpy_img', sharpy_img)
#cv2.imshow('sharpxy_img', sharpxy_img)
#key = cv2.waitKey()

#角点检测
img = cv2.imread('D:\Artificial_Intelligence\pic\AGJR7725.JPG')
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
img_harris = cv2.cornerHarris(img_gray, 2, 3, 0.05)  #blockSize：角点检测中考虑的区域大小 ksize：Sobel求导中使用的窗口大小 k：Harris 角点检测方程中的自由参数，取值参数为 [0.04 0.06]
img_harris = cv2.dilate(img_harris , None)  #膨胀操作，将角点区域扩大
thres = 0.05 * np.max(img_harris)
img[img_harris > thres] = [0, 0, 255]
cv2.imshow('img_harris ', img)
key = cv2.waitKey()

#sift操作
sift = cv2.xfeatures2d.SIFT_create()  #创建sift
kp = sift.detect(img,None)   #Kp返回的特征点是一个带有很多不用属性的特殊结构体，属性当中有坐标，方向、角度等等
img2=cv2.drawKeypoints(img,kp,img) #显示特征点位置
cv2.imshow('img_kp',img2)
# compute SIFT descriptor
kp,des = sift.compute(img,kp) #kp表示输入的关键点，dst表示输出的sift特征向量，通常是128维的)
img_sift= cv2.drawKeypoints(img,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #显示特征点位置，强度，方向
cv2.imshow('img_sift',img_sift)
cv2.waitKey(0)