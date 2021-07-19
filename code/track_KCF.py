from imutils.video import FPS
import imutils
import cv2
 
 
tracker = cv2.TrackerKCF_create()

initBB = None

filename = '/Users/loujieming/小铭不熬夜/作业和笔记/图像工程/图片视频/viplane.avi'
vs = cv2.VideoCapture(filename)
# initialize the FPS throughput estimator
fps = None
# loop over frames from the video stream

while (True):
    _,frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        # update the FPS counter     
        fps.update()
        fps.stop()
       
    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break
        
vs.release()
cv2.destroyAllWindows()

