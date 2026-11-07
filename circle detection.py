import cv2
import numpy as np
eyes = cv2.imread("object detection/images/eyes.jpg",0)
greyblurred = cv2.blur(eyes,(3,3))
detected_circles = cv2.HoughCircles(greyblurred,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=1,maxRadius=40)
if detected_circles is not None:
    detected_circles=np.uint16(np.around(detected_circles))
    for pt in detected_circles[0,:]:
        x = pt[0]
        y= pt[1]
        radius = pt[2]
        cv2.circle(eyes,(x,y),radius,(20,50,180),2)
        cv2.imshow("eyes",eyes)
cv2.waitKey(0)