import cv2
import numpy as np

# Load the image
img = cv2.imread("D:/Work/Programming/Python project/Projects/Nakshastra/StereoCam/images/test.jpeg")

# Color-segmentation to get binary mask
# lwr = np.array([0, 0, 143])
# upr = np.array([179, 61, 252])
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# msk = cv2.inRange(hsv, lwr, upr)

# Extract chess-board
# krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
# dlt = cv2.dilate(img, krn, iterations=5)
# res = 255 - cv2.bitwise_and(dlt, img)

# Displaying chess-board features
# res = np.uint8(res)
img = cv2.resize(img, (1280, 720))
ret, corners = cv2.findChessboardCorners(img, (7, 7), None)
if ret:
    print(corners)
    fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
    cv2.imshow("fnl", fnl)
    cv2.waitKey(0)
else:
    print("Chessboard not found")
    cv2.imshow("res", img)
    cv2.waitKey(0)