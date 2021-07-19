import cv2 as cv
import copy
import matplotlib.pyplot as plt
import numpy as np

def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape#获取shape的数值，height和width、通道
 
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下
    cv.imshow("con-bri-demo", dst)
    return dst

# 打开图像
filename = r'/Users/loujieming/Downloads/rice.JPG'
image = cv.imread(filename)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 修改图像对比度


# 大津算法灰度阈值化
# thr, bw = cv.threshold(gray,129,200,cv.THRESH_BINARY)
thr, bw = cv.threshold(gray, 0, 0xff, cv.THRESH_OTSU)
print('Threshold is :', thr)

# 画出灰度直方图
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
bw = cv.morphologyEx(bw, cv.MORPH_OPEN, element)

seg = copy.deepcopy(bw)
# 计算轮廓
cnts, hier = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
minarea = 5000
maxarea = 0
meanarea = 0
count = 0
# 遍历所有区域，并去除面积过小的
for i in range(len(cnts), 0, -1):
    c = cnts[i-1]
    area = cv.contourArea(c)
    maxarea = max(maxarea,area)
    minarea = min(minarea,area)
    meanarea += area
    if area < 10:
        continue
    count = count + 1
    print("blob", i, " : ", area)

    # 区域画框并标记
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0xff), 1)
    cv.putText(image, str(count), (x, y), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0xff, 0))

meanarea /= len(cnts)
print("米粒数量： ", count)
print("maxarea=" + str(maxarea))
print("minarea=" + str(minarea))
print("meanarea=" + str(meanarea))

cv.startWindowThread()

cv.imshow("源图", image)
cv.imwrite("源图.jpg",image)
cv.waitKey()
cv.imshow("阈值化图", bw)
cv.imwrite("阈值化图.jpg",bw)
cv.waitKey()
cv.destroyAllWindows()