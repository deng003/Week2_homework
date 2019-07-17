import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':

    #初始化照片，img1和img2高度相同，img2的宽度小于img1
    top, bot, left, right = 100, 100, 0, 700 #进行投影变换时，右图会向右缩，所以需要在右侧增加padding，以完整显示图片
    img1 = cv.imread('D:\Python\CV_course_V2\pic\image_01.jpg')
    img2 = cv.imread('D:\Python\CV_course_V2\pic\image_02.jpg')
    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)

    #采用sift算法进行关键点匹配
    sift = cv.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)     #kp1返回ID，里面包含信息：特征点坐标，邻域直径，方向（360内），强度，所在金字塔的组，聚类ID
    kp2, des2 = sift.detectAndCompute(img2gray, None)     #descriptors：它是对 keypoints 进一步处理的结果。通常它具有更低的维度，从而使得图像块能够在另一幅不同的图像中被更快地识别
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # 指定算法为随机KD树算法，采用5棵树进行计算，默认为4
    search_params = dict(checks=50)                            #迭代次数50次
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)                 #des1原图 查找目标点最近的K个数据，k数值越高精度越高
                                                              # matchs返回信息包含（原图ID，匹配训练图ID，二者之间欧式距离），由于K=2，故一个原图ID对应2个训练图ID
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches): #遍历对象，如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留  提取优秀的特征点
        if m.distance < 0.7*n.distance:
            good.append(m)

    #根据特征点匹配，进行图形拼接
    rows, cols = srcImg.shape[:2]   #提取原图像行和列
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:  #至少4个点以上才可以进行投影变换，10个点更好
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])  #将选择的原图匹配点坐标存储于src_pts数组中中
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])  #将选择的训练图匹配点坐标存储于dst_pts数组中
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0) # 生成变换矩阵
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]), flags=cv.WARP_INVERSE_MAP) #将训练图进行根据变换矩阵进行透视变换，生成新图warpImg

        # 寻找新图和原图的重叠列
        print(srcImg.shape,warpImg.shape)
        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col #找出重叠部分的左列
                break
        for col in range(cols-1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col #找出重叠部分的右列
                break

        # 创建拼接图res，以存储原图和新图的拼接结果，为防止图形溢出，图形宽度增加700（也可为其他数字）
        res = np.zeros([rows, cols+700, 3], np.uint8)
        for row in range(0, rows):
            #将重叠部分之前的图形存储于res图中
            for col in range(0, right):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col] #如果srcImg中为空，将对应的warpImg像素赋予res中
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]  #如果warpImg中为空，将对应的srcImg像素赋予res中
                # 将重叠部分根据距离比例存储于res图中
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)
            # 将重叠部分之后生成的新图存储于res图中
            for col in range(right, warpImg.shape[1]):
                res[row, col] = warpImg[row, col]
        cv.imshow('res',res)
        cv.imwrite('D:\Python\CV_course_V2\pic/res.jpg',res)
        key = cv.waitKey()
    else:
        #如果找不到足够的匹配点，输出提醒
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
