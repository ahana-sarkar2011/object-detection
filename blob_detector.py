import cv2
import numpy as np
blobs = cv2.imread("object detection/images/blobs.jpg",0)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(blobs)
blank = np.zeros((1,1))
blobs = cv2.putText(blobs,"number of circles ="+str(len(keypoints)),(20,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(20,20,250))
blobs = cv2.drawKeypoints(blobs,keypoints,blank,(20,20,250),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("blobs",blobs)
cv2.waitKey(0)