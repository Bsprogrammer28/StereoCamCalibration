import cv2
import numpy as np

stereo_PARAMS = np.load('stereo_camera_calibration.npz')
cameraMatrix1 = stereo_PARAMS['CM1']
cameraMatrix2 = stereo_PARAMS['CM2']
distCoeffs1 = stereo_PARAMS['dist1']
distCoeffs2 = stereo_PARAMS['dist2']
R = stereo_PARAMS['R']
T = stereo_PARAMS['T']

imgL = cv2.imread("images\synched\L_2.png")
imgR = cv2.imread("images\synched\R_2.png")
imageSize = imgL.shape[:2][::-1]  # (width, height)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1  # 0=crop, 1=full image, -1=automatic
)
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)

rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(grayL, grayR)

disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def draw_lines(img):
    for y in range(0, img.shape[0], 20):
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    return img

cv2.imshow("Left Rectified", draw_lines(rectifiedL.copy()))
cv2.imshow("Right Rectified", draw_lines(rectifiedR.copy()))
cv2.imshow("Disparity Map", disp_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

