import cv2
import sys
import numpy as np

filename = '/Users/loujieming/小铭不熬夜/作业和笔记/图像工程/图片视频/video1.avi'
video = cv2.VideoCapture(filename)

ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

bbox = (287, 23, 86, 320)

bbox1 = cv2.selectROI(frame, False)
bbox2 = cv2.selectROI(frame, False)
print(bbox1, bbox2)

tracker1 = cv2.TrackerKCF_create()
tracker2 = cv2.TrackerKCF_create()
ok1 = tracker1.init(frame, bbox1)
ok2 = tracker2.init(frame, bbox2)

p31 = (int(bbox1[0] + bbox1[2]/2), int(bbox1[1] + bbox1[3]/2))
p32 = (int(bbox2[0] + bbox2[2]/2), int(bbox2[1] + bbox2[3]/2))
list1 = [p31]
list2 = [p32]

while True:
    # Read a new frame
    ok, frame = video.read()
    mask1 = np.zeros_like(frame)                         # 为绘制创建掩码图片
    mask2 = np.zeros_like(frame) 
    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok1, bbox1 = tracker1.update(frame)
    ok2, bbox2 = tracker2.update(frame)
    print(bbox1, bbox2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok1:
        # Tracking success
        p1 = (int(bbox1[0]), int(bbox1[1]))
        p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
        p0 = (int(bbox1[0] + bbox1[2]/2), int(bbox1[1] + bbox1[3]/2))
        frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        list1.append(p0)
        for i in range(len(list1)-1):
            mask1 = cv2.line(mask1,list1[i],list1[i+1],(0,255,0),1)

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    if ok2:
        # Tracking success
        p1 = (int(bbox2[0]), int(bbox2[1]))
        p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        p0 = (int(bbox2[0] + bbox2[2]/2), int(bbox2[1] + bbox2[3]/2))
        frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        list2.append(p0)
        print(list2)
        for i in range(len(list2)-1):
            mask2 = cv2.line(mask2,list2[i],list2[i+1],(0,255,0),1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    img = cv2.add(frame, mask1)
    img = cv2.add(img,mask2)

    cv2.imshow("Tracking", img)

    # Exit if ESC pressed
    k = cv2.waitKey(100) & 0xff
    if k == 27: break
    elif k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

