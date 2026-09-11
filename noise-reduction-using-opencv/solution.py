import cv2
import matplotlib.pyplot as plt
import numpy as np
#print(cv2.__version__)

#image = cv2.imread('demo1.jpg')
def CannyEdge(image):
    imagegray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.GaussianBlur(imagegray, (5, 5), 0)
    cannyimage = cv2.Canny(imagegray, 60, 180)
    return cannyimage

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(100, height),(325, 25),(400, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

image = cv2.imread('demo1.jpg')
#cv2.imshow('gray image', imagegray)
#cv2.imshow('canny image', CannyEdge(image))
plt.imshow(CannyEdge(image))
#plt.imshow(cannyimage)
plt.show()
#cap = cv2.VideoCapture("test.mp4")
#while(cap.isOpened()):
#    _, frame = cap.read()
image = cv2.imread('demo1.jpg')
canny = CannyEdge(image)
cropped_Image = region_of_interest(canny)
rho = 2
theta = np.pi/180
threshold = 100
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([ ]), minLineLength=40, maxLineGap=5)
line_image = display_lines(image, lines)
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
cv2.imshow("Image", combo_image)
#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break
#cv2.release(0)
cv2.destroyAllWindows()