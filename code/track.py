import numpy as np
import cv2 as cv

cap = cv.VideoCapture('/Users/loujieming/小铭不熬夜/作业和笔记/图像工程/图片视频/Cap02t3.avi')

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# 光流法参数
# maxLevel 未使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机生成的颜色
color = np.random.randint(0, 255, (100, 3))


ret, old_frame = cap.read()                             # 取出视频的第一帧
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)  # 灰度化
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params) # 获得旧的关注点
mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片

while True:
    _, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 计算光流以获取点的新位置
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选择good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # 绘制跟踪框
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 1)
        frame = cv.circle(frame, (int(a), int(b)), 10, color[i].tolist())

    img = cv.add(frame, mask)
    cv.imshow('frame', img)
   
    k = cv.waitKey(50)  # & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()

