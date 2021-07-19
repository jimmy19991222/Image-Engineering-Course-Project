import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/loujieming/小铭不熬夜/作业和笔记/图像工程/图片视频/Cap02t3.avi')

#测试用,查看视频size
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))

# 构建椭圆结果
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    # 读取视频流
    grabbed, frame = cap.read()

    if frame is None:
        break
    
    # 对帧进行预处理，>>转灰度图>>高斯滤波（降噪：摄像头震动、光照变化）。
    gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    if background is None:
        background = gray_lwpCV
        continue

    # 对比背景之后的帧与背景之间的差异，并得到一个差分图（different map）。
    # 阈值（二值化处理）>>膨胀（dilate）得到图像区域块
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)

    _, fgmask = cv2.threshold(diff.copy(), 0, 0xff, cv2.THRESH_BINARY)

     # 显示矩形框：计算一幅图像中目标的轮廓
    cnts, hier  = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有区域，并去除面积过小的
    for i in range(len(cnts), 0, -1):
        c = cnts[i-1]
        area = cv2.contourArea(c)
        if area < 600:
            continue

        # 区域画框并标记
        x, y, w, h = cv2.boundingRect(c)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('contours', frame)
    # cv2.imshow('dis', diff)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):    # 按'q'健退出循环
        break
# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

