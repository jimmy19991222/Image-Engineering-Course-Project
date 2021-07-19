import cv2 as cv
import numpy as np

# 转灰
def rgb2gray(img):
    h=img.shape[0]
    w=img.shape[1]
    img1=np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            img1[i,j]=0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,1]
    return img1

# 局部大津算法实现
def otsu(img):
    h=img.shape[0]
    w=img.shape[1]
    otsuimg=np.zeros((h,w),np.uint8)
    for i in range(w):   # 遍历列
        sigma=threshold=0   # 定义类间方差和最终阈值
        histogram=np.zeros(256,np.int32)   # 初始化各灰度级个数统计
        probability=np.zeros(256,np.float32)   # 初始化各灰度级概率分布
        for j in range (h):   # 遍历行，进行otsu算法
            s=img[j,i]
            histogram[s]+=1   # 统计灰度级中每个像素在整幅图像中的个数
        for k in range (256):
            probability[k]=histogram[k]/h   # 统计每个灰度级占图像中的分布
        for p in range (255):
            w0 = w1 = 0   # 定义前景像素点和背景像素点灰度级占图像中的分布
            fgs = bgs = 0   # 定义前景像素点灰度级总和and背景像素点灰度级总和
            for q in range (256):
                if q<=p:   # 当前i为分割阈值
                    w0+=probability[q]   # 前景像素点占整幅图像的比例累加
                    fgs+=q*probability[q]   # 前景像素点的平均灰度
                else:
                    w1+=probability[q]   # 背景像素点占整幅图像的比例累加
                    bgs+=q*probability[q]   # 背景像素点的平均灰度
            u0=fgs/w0
            u1=bgs/w1
            g=w0*w1*(u0-u1)**2   # 类间方差
            if g>=sigma:
                sigma=g
                threshold=p
        for j in range (h):   # 对某列的每一行进行二值化
            if img[j,i]>threshold:
                otsuimg[j,i]=255
            else:
                otsuimg[j,i]=0
    return otsuimg

image = cv.imread(r'/Users/loujieming/Downloads/rice.JPG')
grayimage = rgb2gray(image)
otsuimage = otsu(grayimage)
cv.imshow("image", image)
cv.imwrite("rice源图.jpg",image)
cv.imshow("otsuimage", otsuimage)
cv.imwrite("otsuimage.jpg", otsuimage)
cv.waitKey(0)
cv.destroyAllWindows()

