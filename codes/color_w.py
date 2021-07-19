import cv2 as cv
import matplotlib.pyplot as plt

filename = r'/Users/loujieming/Downloads/season.JPG'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

plt.figure()
plt.subplot(2,4,1)
plt.imshow(img[:, :, [2, 1, 0]]) # 转化成RGB
plt.title("img")
plt.subplot(2,4,2)
plt.imshow(img[:, :, [2, 1, 0]][:, :, 0])
plt.title("Blue")
plt.subplot(2,4,3)
plt.imshow(img[:, :, [2, 1, 0]][:, :, 1])
plt.title("Green")
plt.subplot(2,4,4)
plt.imshow(img[:, :, [2, 1, 0]][:, :, 2])
plt.title("Red")

plt.subplot(2,4,5)
plt.imshow(hsv[:, :, [2, 1, 0]])
plt.title("hsv")
plt.subplot(2,4,6)
plt.imshow(hsv[:, :, [2, 1, 0]][:, :, 0])
plt.title("Hue")
plt.subplot(2,4,7)
plt.imshow(hsv[:, :, [2, 1, 0]][:, :, 1])
plt.title("Saturation")
plt.subplot(2,4,8)
plt.imshow(hsv[:, :, [2, 1, 0]][:, :, 2])
plt.title("Value")

plt.show()

cv.imshow('source image', img)
cv.imwrite("source image.jpg",img)
cv.imwrite("hsv.jpg",hsv)
cv.imshow('gray', gray)
cv.waitKey()

cv.imshow("Hue", hsv[:, :, 0])
cv.imshow("Saturation", hsv[:, :, 1])
cv.imshow("Value", hsv[:, :, 2])
cv.imwrite('Hue.jpg', hsv[:,:,0])
cv.imwrite('Saturation.jpg', hsv[:,:,1])
cv.imwrite('Value.jpg', hsv[:,:,2])
cv.waitKey()

cv.imshow("Blue", img[:, :, 0])
cv.imshow("Green", img[:, :, 1])
cv.imshow("Red", img[:, :, 2])
cv.imwrite('Blue.jpg', img[:,:,0])
cv.imwrite('Green.jpg', img[:,:,1])
cv.imwrite('Red.jpg', img[:,:,2])

cv.waitKey()
cv.destroyAllWindows()